import torch
import triton
import triton.language as tl


DEFAULT_GEMM1_SILU_CFG = {
    "name": "g1_BM64_BN128_BK32_GM8_W4_S2",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

DEFAULT_GEMM2_COMBINE_CFG = {
    "name": "g2_BM64_BN256_BK32_GM8_W8_S3",
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}


@triton.jit
def _moe_gemm1_silu_kernel(
    x_ptr,
    w13_ptr,
    hidden_ptr,
    expert_offsets_ptr,
    tile_expert_ids_ptr,
    tile_m_ids_ptr,
    tile_n_ids_ptr,
    K,
    I,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_hm,
    stride_hi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)

    expert_id = tl.load(tile_expert_ids_ptr + tile_id)
    pid_m = tl.load(tile_m_ids_ptr + tile_id)
    pid_n = tl.load(tile_n_ids_ptr + tile_id)

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_m = expert_start + offs_m

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x_ptrs = (
            x_ptr
            + global_m[:, None] * stride_xm
            + k_idxs[None, :] * stride_xk
        )

        # w13 shape: [E, K, 2I]
        # gate: w13[e, :, 0:I]
        # up  : w13[e, :, I:2I]
        w_gate_ptrs = (
            w13_ptr
            + expert_id * stride_we
            + k_idxs[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )

        w_up_ptrs = (
            w13_ptr
            + expert_id * stride_we
            + k_idxs[:, None] * stride_wk
            + (offs_n[None, :] + I) * stride_wn
        )

        x_mask = (offs_m[:, None] < expert_m) & (k_idxs[None, :] < K)
        w_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < I)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_gate = tl.load(w_gate_ptrs, mask=w_mask, other=0.0)
        w_up = tl.load(w_up_ptrs, mask=w_mask, other=0.0)

        acc_gate += tl.dot(x, w_gate)
        acc_up += tl.dot(x, w_up)

    hidden = acc_gate * tl.sigmoid(acc_gate) * acc_up

    hidden_ptrs = (
        hidden_ptr
        + global_m[:, None] * stride_hm
        + offs_n[None, :] * stride_hi
    )

    out_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < I)

    tl.store(hidden_ptrs, hidden, mask=out_mask)


def moe_gemm1_silu(
    x_sorted: torch.Tensor,
    w13: torch.Tensor,
    hidden: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_expert_ids: torch.Tensor | None = None,
    tile_m_ids: torch.Tensor | None = None,
    tile_n_ids: torch.Tensor | None = None,
    cfg: dict | None = None,
):
    """
    x_sorted:
        [T, H]

    w13:
        [E, H, 2I]

    hidden:
        [T, I]

    expert_offsets:
        [E + 1]

    实现:
        gate = x_sorted @ w13[e, :, :I]
        up   = x_sorted @ w13[e, :, I:]
        hidden = silu(gate) * up
    """

    assert x_sorted.is_cuda
    assert w13.is_cuda
    assert hidden.is_cuda
    assert expert_offsets.is_cuda

    assert x_sorted.dim() == 2
    assert w13.dim() == 3
    assert hidden.dim() == 2
    assert expert_offsets.dim() == 1

    T, H = x_sorted.shape
    E, H_w, two_I = w13.shape
    T_h, I = hidden.shape

    assert H == H_w, \
        f"x_sorted.shape={x_sorted.shape}, w13.shape={w13.shape}"

    assert two_I == 2 * I, \
        f"w13 last dim must be 2 * hidden dim, got w13={two_I}, hidden={I}"

    assert T == T_h, \
        f"x_sorted and hidden token dim mismatch: {T} vs {T_h}"

    assert expert_offsets.shape == (E + 1,)
    assert int(expert_offsets[0].item()) == 0
    assert int(expert_offsets[-1].item()) == T

    assert x_sorted.dtype in (torch.float16, torch.bfloat16)
    assert w13.dtype == x_sorted.dtype
    assert hidden.dtype == x_sorted.dtype

    assert w13.is_contiguous(), \
        "w13 must be contiguous with shape [E, H, 2I]"

    cfg = DEFAULT_GEMM1_SILU_CFG if cfg is None else cfg

    if tile_expert_ids is None or tile_m_ids is None or tile_n_ids is None:
        import fray

        # 注意：这里 metadata 的 N 是 I，不是 2I。
        tile_expert_ids, tile_m_ids, tile_n_ids = fray.triton.build_grouped_gemm_metadata(
            expert_offsets,
            I,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            GROUP_M=cfg["GROUP_M"],
        )

    assert tile_expert_ids.is_cuda
    assert tile_m_ids.is_cuda
    assert tile_n_ids.is_cuda

    total_tiles = tile_expert_ids.numel()

    if total_tiles == 0:
        return hidden

    grid = (total_tiles,)

    _moe_gemm1_silu_kernel[grid](
        x_sorted,
        w13,
        hidden,
        expert_offsets,
        tile_expert_ids,
        tile_m_ids,
        tile_n_ids,
        H,
        I,
        x_sorted.stride(0),
        x_sorted.stride(1),
        w13.stride(0),
        w13.stride(1),
        w13.stride(2),
        hidden.stride(0),
        hidden.stride(1),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    return hidden

@triton.jit
def _moe_gemm2_combine_persistent_kernel(
    hidden_ptr,
    w2_ptr,
    sorted_token_ids_ptr,
    sorted_token_weights_ptr,
    output_ptr,
    expert_offsets_ptr,
    tile_expert_ids_ptr,
    tile_m_ids_ptr,
    tile_n_ids_ptr,
    TOTAL_TILES,
    K,
    N,
    stride_hm,
    stride_hk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # chunked persistent scheduling:
    tiles_per_program = tl.cdiv(TOTAL_TILES, num_programs)
    tile_start = pid * tiles_per_program
    tile_end = tl.minimum(tile_start + tiles_per_program, TOTAL_TILES)

    for tile_id in tl.range(tile_start, tile_end):
        expert_id = tl.load(tile_expert_ids_ptr + tile_id)
        pid_m = tl.load(tile_m_ids_ptr + tile_id)
        pid_n = tl.load(tile_n_ids_ptr + tile_id)

        expert_start = tl.load(expert_offsets_ptr + expert_id)
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        expert_m = expert_end - expert_start

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        global_m = expert_start + offs_m

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k

            hidden_ptrs = (
                hidden_ptr
                + global_m[:, None] * stride_hm
                + k_idxs[None, :] * stride_hk
            )

            w2_ptrs = (
                w2_ptr
                + expert_id * stride_we
                + k_idxs[:, None] * stride_wk
                + offs_n[None, :] * stride_wn
            )

            hidden_mask = (offs_m[:, None] < expert_m) & (k_idxs[None, :] < K)
            w2_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < N)

            hidden = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0)
            w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

            acc += tl.dot(hidden, w2)

        token_ids = tl.load(
            sorted_token_ids_ptr + global_m,
            mask=offs_m < expert_m,
            other=0,
        )

        token_weights = tl.load(
            sorted_token_weights_ptr + global_m,
            mask=offs_m < expert_m,
            other=0.0,
        ).to(tl.float32)

        vals = acc * token_weights[:, None]

        output_ptrs = (
            output_ptr
            + token_ids[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )

        out_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < N)

        if USE_ATOMIC:
            tl.atomic_add(
                output_ptrs,
                vals,
                mask=out_mask,
                sem="relaxed",
            )
        else:
            tl.store(
                output_ptrs,
                vals,
                mask=out_mask,
            )    

def _get_num_sms(tensor: torch.Tensor):
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    return torch.cuda.get_device_properties(device_index).multi_processor_count


def moe_gemm2_combine(
    hidden: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_token_weights: torch.Tensor,
    output: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_expert_ids: torch.Tensor | None = None,
    tile_m_ids: torch.Tensor | None = None,
    tile_n_ids: torch.Tensor | None = None,
    cfg: dict | None = None,
    use_atomic: bool = True,
    persistent: bool = False,
    persistent_waves: int = 2,
):
    assert hidden.is_cuda
    assert w2.is_cuda
    assert sorted_token_ids.is_cuda
    assert sorted_token_weights.is_cuda
    assert output.is_cuda
    assert expert_offsets.is_cuda

    assert hidden.dim() == 2
    assert w2.dim() == 3
    assert output.dim() == 2
    assert expert_offsets.dim() == 1
    assert sorted_token_ids.dim() == 1
    assert sorted_token_weights.dim() == 1

    T, I = hidden.shape
    E, I_w, H = w2.shape
    num_tokens, H_out = output.shape

    assert I == I_w
    assert H == H_out
    assert expert_offsets.shape == (E + 1,)
    assert int(expert_offsets[0].item()) == 0
    assert int(expert_offsets[-1].item()) == T

    assert sorted_token_ids.shape == (T,)
    assert sorted_token_weights.shape == (T,)

    assert sorted_token_ids.dtype in (torch.int32, torch.int64)
    assert sorted_token_weights.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )

    assert hidden.dtype in (torch.float16, torch.bfloat16)
    assert w2.dtype == hidden.dtype
    assert output.dtype == hidden.dtype

    assert w2.is_contiguous(), "w2 must be contiguous with shape [E, I, H]"

    cfg = DEFAULT_GEMM2_COMBINE_CFG if cfg is None else cfg

    if tile_expert_ids is None or tile_m_ids is None or tile_n_ids is None:
        import fray

        tile_expert_ids, tile_m_ids, tile_n_ids = fray.triton.build_grouped_gemm_metadata(
            expert_offsets,
            H,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            GROUP_M=cfg["GROUP_M"],
        )

    total_tiles = tile_expert_ids.numel()

    if total_tiles == 0:
        return output

    if persistent:
        num_sms = _get_num_sms(hidden)
        num_programs = min(total_tiles, num_sms * persistent_waves)
        grid = (num_programs,)

        _moe_gemm2_combine_persistent_kernel[grid](
            hidden,
            w2,
            sorted_token_ids,
            sorted_token_weights,
            output,
            expert_offsets,
            tile_expert_ids,
            tile_m_ids,
            tile_n_ids,
            total_tiles,
            I,
            H,
            hidden.stride(0),
            hidden.stride(1),
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            USE_ATOMIC=use_atomic,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    else:
        grid = (total_tiles,)

        _moe_gemm2_combine_kernel[grid](
            hidden,
            w2,
            sorted_token_ids,
            sorted_token_weights,
            output,
            expert_offsets,
            tile_expert_ids,
            tile_m_ids,
            tile_n_ids,
            I,
            H,
            hidden.stride(0),
            hidden.stride(1),
            w2.stride(0),
            w2.stride(1),
            w2.stride(2),
            output.stride(0),
            output.stride(1),
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            USE_ATOMIC=use_atomic,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output
    

@triton.jit
def _moe_gemm2_combine_kernel(
    hidden_ptr,
    w2_ptr,
    sorted_token_ids_ptr,
    sorted_token_weights_ptr,
    output_ptr,
    expert_offsets_ptr,
    tile_expert_ids_ptr,
    tile_m_ids_ptr,
    tile_n_ids_ptr,
    K,
    N,
    stride_hm,
    stride_hk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_ATOMIC: tl.constexpr = True,
):
    tile_id = tl.program_id(0)

    expert_id = tl.load(tile_expert_ids_ptr + tile_id)
    pid_m = tl.load(tile_m_ids_ptr + tile_id)
    pid_n = tl.load(tile_n_ids_ptr + tile_id)

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    expert_m = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_m = expert_start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        hidden_ptrs = (
            hidden_ptr
            + global_m[:, None] * stride_hm
            + k_idxs[None, :] * stride_hk
        )

        # w2 shape: [E, I, H]
        w2_ptrs = (
            w2_ptr
            + expert_id * stride_we
            + k_idxs[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )

        hidden_mask = (offs_m[:, None] < expert_m) & (k_idxs[None, :] < K)
        w2_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < N)

        hidden = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0)
        w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        acc += tl.dot(hidden, w2)

    token_ids = tl.load(
        sorted_token_ids_ptr + global_m,
        mask=offs_m < expert_m,
        other=0,
    )

    token_weights = tl.load(
        sorted_token_weights_ptr + global_m,
        mask=offs_m < expert_m,
        other=0.0,
    ).to(tl.float32)

    vals = acc * token_weights[:, None]

    output_ptrs = (
        output_ptr
        + token_ids[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )

    out_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < N)

    if USE_ATOMIC:
        tl.atomic_add(
            output_ptrs,
            vals,
            mask=out_mask,
            sem="relaxed",
        )
    else:
        tl.store(
            output_ptrs,
            vals,
            mask=out_mask,
        )


def _build_metadata(
    expert_offsets: torch.Tensor,
    N: int,
    cfg: dict,
):
    import fray

    return fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        N,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        GROUP_M=cfg["GROUP_M"],
    )


def fused_moe(
    x_sorted: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_token_weights: torch.Tensor,
    output: torch.Tensor | None = None,
    num_tokens: int | None = None,
    gemm1_cfg: dict | None = None,
    gemm2_cfg: dict | None = None,
    gemm1_metadata=None,
    gemm2_metadata=None,
    use_atomic: bool = True,
    clear_output: bool = True,
    persistent_gemm2: bool = False,
    persistent_waves: int = 2,
    hidden: torch.Tensor | None = None,
):
    assert x_sorted.is_cuda
    assert w13.is_cuda
    assert w2.is_cuda
    assert expert_offsets.is_cuda
    assert sorted_token_ids.is_cuda
    assert sorted_token_weights.is_cuda

    assert x_sorted.dim() == 2
    assert w13.dim() == 3
    assert w2.dim() == 3
    assert expert_offsets.dim() == 1
    assert sorted_token_ids.dim() == 1
    assert sorted_token_weights.dim() == 1

    T, H = x_sorted.shape
    E, H_w13, two_I = w13.shape
    E_w2, I, H_w2 = w2.shape

    assert H == H_w13
    assert E == E_w2
    assert two_I == 2 * I
    assert H == H_w2

    assert expert_offsets.shape == (E + 1,)
    assert int(expert_offsets[0].item()) == 0
    assert int(expert_offsets[-1].item()) == T

    assert sorted_token_ids.shape == (T,)
    assert sorted_token_weights.shape == (T,)

    assert x_sorted.dtype in (torch.float16, torch.bfloat16)
    assert w13.dtype == x_sorted.dtype
    assert w2.dtype == x_sorted.dtype

    assert w13.is_contiguous(), "w13 must be contiguous with shape [E, H, 2I]"
    assert w2.is_contiguous(), "w2 must be contiguous with shape [E, I, H]"

    if num_tokens is None:
        num_tokens = int(sorted_token_ids.max().item()) + 1

    if output is None:
        output = torch.empty(
            (num_tokens, H),
            device=x_sorted.device,
            dtype=x_sorted.dtype,
        )

    assert output.is_cuda
    assert output.shape == (num_tokens, H)
    assert output.dtype == x_sorted.dtype

    gemm1_cfg = DEFAULT_GEMM1_SILU_CFG if gemm1_cfg is None else gemm1_cfg
    gemm2_cfg = DEFAULT_GEMM2_COMBINE_CFG if gemm2_cfg is None else gemm2_cfg

    # top_k > 1: atomic_add，需要先清零。
    # top_k == 1: store 覆盖写，可以跳过清零。
    if clear_output:
        output.zero_()

    if hidden is None:
        hidden = torch.empty(
            (T, I),
            device=x_sorted.device,
            dtype=x_sorted.dtype,
        )
    else:
        assert hidden.is_cuda
        assert hidden.shape == (T, I), \
            f"hidden workspace shape must be ({T}, {I}), got {hidden.shape}"
        assert hidden.dtype == x_sorted.dtype
    
    if gemm1_metadata is None:
        import fray

        gemm1_metadata = fray.triton.build_grouped_gemm_metadata(
            expert_offsets,
            I,
            BLOCK_M=gemm1_cfg["BLOCK_M"],
            BLOCK_N=gemm1_cfg["BLOCK_N"],
            GROUP_M=gemm1_cfg["GROUP_M"],
        )

    if gemm2_metadata is None:
        import fray

        gemm2_metadata = fray.triton.build_grouped_gemm_metadata(
            expert_offsets,
            H,
            BLOCK_M=gemm2_cfg["BLOCK_M"],
            BLOCK_N=gemm2_cfg["BLOCK_N"],
            GROUP_M=gemm2_cfg["GROUP_M"],
        )

    moe_gemm1_silu(
        x_sorted,
        w13,
        hidden,
        expert_offsets,
        gemm1_metadata[0],
        gemm1_metadata[1],
        gemm1_metadata[2],
        cfg=gemm1_cfg,
    )

    moe_gemm2_combine(
        hidden,
        w2,
        sorted_token_ids,
        sorted_token_weights,
        output,
        expert_offsets,
        gemm2_metadata[0],
        gemm2_metadata[1],
        gemm2_metadata[2],
        cfg=gemm2_cfg,
        use_atomic=use_atomic,
        persistent = persistent_gemm2,
        persistent_waves=persistent_waves
    )

    return output