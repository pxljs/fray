import torch
import triton
import triton.language as tl

@triton.jit
def _grouped_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    expert_offsets_ptr,
    tile_expert_ids_ptr,
    tile_m_ids_ptr,
    tile_n_ids_ptr,
    K,
    N,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
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
    num_experts = expert_end - expert_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    global_m = expert_start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_idxs = k + offs_k

        x_ptrs = x_ptr + global_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk
        w_ptrs = w_ptr + expert_id * stride_we + k_idxs[:, None] * stride_wk + offs_n[None, :] * stride_wn

        x_mask = (offs_m[:, None] < num_experts) & (k_idxs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_idxs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)
    
    out_ptrs = out_ptr + global_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < num_experts) & (offs_n[None, :] < N)

    tl.store(out_ptrs, acc, mask=out_mask)

def _get_num_warps(block_m: int, block_n: int, block_k: int) -> int:
    tile = block_m * block_n

    if tile <= 64 * 64:
        return 4
    else:
        return 8
    

def build_grouped_gemm_metadata(
    expert_offsets: torch.Tensor,
    N: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 128,
    GROUP_M: int = 8,
):
    assert expert_offsets.is_cuda
    assert expert_offsets.dim() == 1
    assert expert_offsets.dtype in (torch.int32, torch.int64)

    device = expert_offsets.device
    num_experts = expert_offsets.numel() - 1

    num_pid_n = triton.cdiv(N, BLOCK_N)

    tokens_per_expert = expert_offsets[1:] - expert_offsets[:-1]

    tiles_m = torch.div(
        tokens_per_expert + BLOCK_M - 1,
        BLOCK_M,
        rounding_mode="floor",
    )

    tile_expert_ids_list = []
    tile_m_ids_list = []
    tile_n_ids_list = []

    tiles_m_cpu = tiles_m.cpu().tolist()

    for e in range(num_experts):
        tm = int(tiles_m_cpu[e])

        if tm == 0:
            continue

        for m_group_start in range(0, tm, GROUP_M):
            group_size_m = min(GROUP_M, tm - m_group_start)

            for pid_n in range(num_pid_n):
                for local_m in range(group_size_m):
                    pid_m = m_group_start + local_m

                    tile_expert_ids_list.append(e)
                    tile_m_ids_list.append(pid_m)
                    tile_n_ids_list.append(pid_n)

    if len(tile_expert_ids_list) == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty, empty

    tile_expert_ids = torch.tensor(
        tile_expert_ids_list,
        device=device,
        dtype=torch.int32,
    )

    tile_m_ids = torch.tensor(
        tile_m_ids_list,
        device=device,
        dtype=torch.int32,
    )

    tile_n_ids = torch.tensor(
        tile_n_ids_list,
        device=device,
        dtype=torch.int32,
    )

    return (
        tile_expert_ids.contiguous(),
        tile_m_ids.contiguous(),
        tile_n_ids.contiguous(),
    )

def grouped_gemm(
    x: torch.Tensor,
    weights: torch.Tensor, 
    output: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_expert_ids: torch.Tensor,
    tile_m_ids: torch.Tensor,
    tile_n_ids: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    GROUP_M: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,    
):
    """
    x:
        [total_tokens, K]

    weights:
        [num_experts, K, N]

    output:
        [total_tokens, N]

    expert_offsets:
        [num_experts + 1]
        expert e 的 token 范围是:
            [expert_offsets[e], expert_offsets[e + 1])
    """

    assert x.is_cuda and weights.is_cuda and output.is_cuda and expert_offsets.is_cuda,\
        "x, weights, output, expert_offsets must be CUDA tensors"
    
    assert x.dim() == 2, \
        "x must be 2D: [total_tokens, K]"

    assert weights.dim() == 3, \
        "weights must be 3D: [num_experts, K, N]"

    assert output.dim() == 2, \
        "output must be 2D: [total_tokens, N]"
    
    total_tokens, K = x.shape
    num_experts, K_w, N = weights.shape

    assert K == K_w, \
        f"K mismatch: x.shape={x.shape}, weights.shape={weights.shape}"

    assert output.shape == (total_tokens, N), \
        f"output.shape must be ({total_tokens}, {N}), got {output.shape}"

    assert expert_offsets.dim() == 1, \
        "expert_offsets must be 1D"

    assert expert_offsets.shape[0] == num_experts + 1, \
        f"expert_offsets shape must be [{num_experts + 1}]"

    assert expert_offsets.dtype in (torch.int32, torch.int64), \
        "expert_offsets must be int32 or int64"

    assert x.dtype in (torch.float16, torch.bfloat16), \
        "x dtype must be fp16 or bf16"
    
    assert int(expert_offsets[0].item()) == 0, \
        "expert_offsets[0] must be 0"

    assert int(expert_offsets[-1].item()) == total_tokens, \
        "expert_offsets[-1] must equal total_tokens"
    

    if tile_expert_ids is None or tile_m_ids is None or tile_n_ids is None:
        tile_expert_ids, tile_m_ids, tile_n_ids = build_grouped_gemm_metadata(
            expert_offsets,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    assert tile_expert_ids.is_cuda and tile_m_ids.is_cuda and tile_n_ids.is_cuda, \
        "tile metadata must be CUDA tensors"
    
    assert tile_expert_ids.dtype == torch.int32
    assert tile_m_ids.dtype == torch.int32
    assert tile_n_ids.dtype == torch.int32

    total_tiles = tile_expert_ids.numel()

    if total_tiles == 0:
        return
    
    grid = (total_tiles,)

    num_warps = _get_num_warps(BLOCK_M, BLOCK_N, BLOCK_K)


    _grouped_gemm_kernel[grid](
        x,
        weights,
        output,
        expert_offsets,
        tile_expert_ids,
        tile_m_ids,
        tile_n_ids,
        K,
        N,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

@triton.jit
def _routed_gemm_kernel(X, W, Y, Tokens, Probabilities, Offsets, Tiles,
                        H: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        ATOMIC: tl.constexpr):
    total = tl.load(Tiles + E)
    for tile in range(tl.program_id(0), total, tl.num_programs(0)):
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), E, tl.int32)
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(Tiles + mid + 1)
            right = tile >= end
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        expert = lo
        local = tile - tl.load(Tiles + expert)
        start = tl.load(Offsets + expert)
        stop = tl.load(Offsets + expert + 1)
        rows = start + (local // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
        cols = (local % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        tokens = tl.load(Tokens + rows, rows < stop, other=0).to(tl.int64)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(H, BK)):
            h = block * BK + inner
            a = tl.load(X + tokens[:, None] * H + h[None, :],
                        (rows[:, None] < stop) & (h[None, :] < H), other=0)
            b = tl.load(W + expert.to(tl.int64) * H * N + h[:, None] * N + cols[None, :],
                        (h[:, None] < H) & (cols[None, :] < N), other=0)
            acc += tl.dot(a, b)
        # Preserve the previous projection rounding before weighting.
        projected = acc.to(X.dtype.element_ty).to(tl.float32)
        probability = tl.load(Probabilities + rows, rows < stop, other=0).to(tl.float32)
        weighted = (projected * probability[:, None]).to(X.dtype.element_ty).to(tl.float32)
        mask = (rows[:, None] < stop) & (cols[None, :] < N)
        if ATOMIC:
            tl.atomic_add(Y + tokens[:, None] * N + cols[None, :], weighted, mask, sem="relaxed")
        else:
            tl.store(Y + rows[:, None].to(tl.int64) * N + cols[None, :], weighted, mask)


@triton.jit
def _gather_combine_kernel(Routed, Positions, Output, N: tl.constexpr,
                           K: tl.constexpr, BLOCK_N: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.full((BLOCK_N,), 0, tl.float32)
    # Fixed route order, independent of dispatch atomics or expert scheduling.
    for route in range(K):
        position = tl.load(Positions + token * K + route)
        contribution = tl.load(Routed + position * N + cols, cols < N, other=0)
        acc += contribution.to(tl.float32)
    tl.store(Output + token * N + cols, acc, cols < N)


def routed_grouped_gemm(x, weights, router_logits, top_k, *, combine="auto"):
    """Inference-only routed projection, using GPU metadata and indirect loads.

    combine: auto uses FP32 atomic accumulation for K <= 4; staged writes
    weighted outputs then gathers/sums in fixed route order. scatter retains
    the previous index_add baseline. All modes return input dtype.
    No host tensor readback. Warm up kernels/device properties before capture.
    """
    from .fused_moe import (
        moe_select_topk_softmax_with_counts, build_moe_dispatch_metadata_fast,
        build_grouped_tile_offsets_no_sync,
    )
    if combine not in ("auto", "atomic", "staged", "scatter"):
        raise ValueError("combine must be auto, atomic, staged or scatter")
    if (x.ndim != 2 or weights.ndim != 3 or router_logits.ndim != 2
            or x.shape[1] != weights.shape[1]
            or router_logits.shape != (x.shape[0], weights.shape[0])):
        raise ValueError("expected x[M,H], weights[E,H,N], router_logits[M,E]")
    if (not x.is_cuda or any(t.device != x.device for t in (weights, router_logits))
            or x.dtype not in (torch.float16, torch.bfloat16) or weights.dtype != x.dtype
            or not x.is_contiguous() or not weights.is_contiguous()):
        raise ValueError("GEMM requires contiguous same-device CUDA FP16/BF16 inputs")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weights, router_logits)):
        raise ValueError("routed_grouped_gemm is inference-only")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= weights.shape[0] <= 16384:
        raise ValueError("top_k must be in [1, E], E <= 16384")
    if router_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("router_logits must be floating point")
    if not x.shape[0]:
        return torch.empty((0, weights.shape[2]), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        ids, probabilities, counts = moe_select_topk_softmax_with_counts(router_logits, top_k)
        atomic = top_k <= 4 if combine == "auto" else combine == "atomic"
        gather = not atomic and combine != "scatter"
        positions = torch.empty(x.shape[0] * top_k, device=x.device, dtype=torch.int64) if gather else None
        token_ids, token_weights, offsets, _ = build_moe_dispatch_metadata_fast(
            ids, probabilities, weights.shape[0], counts=counts, route_positions=positions)
        n = weights.shape[2]
        if not n or not x.shape[1]:
            return torch.zeros((x.shape[0], n), device=x.device, dtype=x.dtype)
        tiles, programs = build_grouped_tile_offsets_no_sync(
            offsets, n, 32, 64, token_ids.numel(), persistent_waves=3)
        output = (torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype) if gather else
                  torch.zeros((x.shape[0], n), device=x.device, dtype=torch.float32))
        routed = output if atomic else torch.empty((token_ids.numel(), n), device=x.device, dtype=x.dtype)
        _routed_gemm_kernel[(programs,)](
            x, weights, routed, token_ids, token_weights, offsets, tiles,
            x.shape[1], n, weights.shape[0], 32, 64, 32, atomic, num_warps=4)
        if gather:
            _gather_combine_kernel[(x.shape[0], triton.cdiv(n, 128))](
                routed, positions, output, n, top_k, 128, num_warps=4)
        elif not atomic:
            output.index_add_(0, token_ids.long(), routed.float())
    return output.to(x.dtype)
