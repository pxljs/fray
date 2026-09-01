import torch
import triton
import triton.language as tl


DEFAULT_GEMM1_PERSISTENT_WAVES_CANDIDATES = (2, 3, 4, 6, 8)
DEFAULT_GEMM2_PERSISTENT_WAVES_CANDIDATES = (1, 2, 3, 4, 6)
LINEAR_EXPERT_DECODE_MAX_EXPERTS = 32
_PERSISTENT_WAVES_CACHE = {}


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
def _moe_zero_int64_kernel(
    ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    tl.store(ptr + offs, tl.zeros((BLOCK_N,), dtype=tl.int64), mask=mask)


def _zero_int64_triton(tensor: torch.Tensor):
    assert tensor.is_cuda and tensor.dtype == torch.int64 and tensor.dim() == 1
    block_n = 1024
    _moe_zero_int64_kernel[(triton.cdiv(tensor.numel(), block_n),)](
        tensor,
        tensor.numel(),
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton.jit
def _moe_zero_tensor_kernel(
    ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    tl.store(ptr + offs, tl.zeros((BLOCK_N,), dtype=tl.float32), mask=mask)


def _zero_tensor_triton(tensor: torch.Tensor):
    assert tensor.is_cuda and tensor.is_contiguous()
    block_n = 1024
    _moe_zero_tensor_kernel[(triton.cdiv(tensor.numel(), block_n),)](
        tensor,
        tensor.numel(),
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton.jit
def _moe_select_topk_softmax_kernel(
    router_logits_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_rm,
    stride_re,
    stride_im,
    stride_ik,
    stride_wm,
    stride_wk,
    BLOCK_E: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < NUM_EXPERTS

    logits = tl.load(
        router_logits_ptr + token_id * stride_rm + expert_offsets * stride_re,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)

    topk_logits = tl.full((BLOCK_TOPK,), -float("inf"), dtype=tl.float32)
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for k in tl.static_range(0, TOP_K):
        max_logit = tl.max(logits, axis=0)
        max_mask = logits == max_logit
        candidate_ids = tl.where(max_mask, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidate_ids, axis=0)

        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            expert_id,
        )
        topk_logits = tl.where(topk_offsets == k, max_logit, topk_logits)
        logits = tl.where(expert_offsets == expert_id, -float("inf"), logits)

    norm_max = tl.max(topk_logits, axis=0)
    topk_exp = tl.exp(topk_logits - norm_max)
    denom = tl.sum(topk_exp, axis=0)
    weights = topk_exp / denom
    tl.store(
        topk_weights_ptr + token_id * stride_wm + topk_offsets * stride_wk,
        weights,
        mask=topk_offsets < TOP_K,
    )


@triton.jit
def _moe_select_topk_softmax_counts_kernel(
    router_logits_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    counts_ptr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_rm,
    stride_re,
    stride_im,
    stride_ik,
    stride_wm,
    stride_wk,
    BLOCK_E: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    token_id = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < NUM_EXPERTS

    logits = tl.load(
        router_logits_ptr + token_id * stride_rm + expert_offsets * stride_re,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)

    topk_logits = tl.full((BLOCK_TOPK,), -float("inf"), dtype=tl.float32)
    topk_offsets = tl.arange(0, BLOCK_TOPK)

    for k in tl.static_range(0, TOP_K):
        max_logit = tl.max(logits, axis=0)
        max_mask = logits == max_logit
        candidate_ids = tl.where(max_mask, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidate_ids, axis=0)

        tl.store(
            topk_ids_ptr + token_id * stride_im + k * stride_ik,
            expert_id,
        )
        tl.atomic_add(counts_ptr + expert_id, 1, sem="relaxed")
        topk_logits = tl.where(topk_offsets == k, max_logit, topk_logits)
        logits = tl.where(expert_offsets == expert_id, -float("inf"), logits)

    norm_max = tl.max(topk_logits, axis=0)
    topk_exp = tl.exp(topk_logits - norm_max)
    denom = tl.sum(topk_exp, axis=0)
    weights = topk_exp / denom
    tl.store(
        topk_weights_ptr + token_id * stride_wm + topk_offsets * stride_wk,
        weights,
        mask=topk_offsets < TOP_K,
    )


def moe_select_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
):
    """
    Select top-k experts per token and normalize selected logits with softmax.

    Args:
        router_logits: [num_tokens, num_experts]
        top_k:         number of experts per token

    Returns:
        topk_ids:     [num_tokens, top_k]
        topk_weights: [num_tokens, top_k]
    """
    assert router_logits.is_cuda
    assert router_logits.dim() == 2
    assert router_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_tokens, num_experts = router_logits.shape
    assert 1 <= top_k <= num_experts

    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, top_k),
            device=router_logits.device,
            dtype=torch.int64,
        )
    else:
        assert topk_ids.is_cuda and topk_ids.shape == (num_tokens, top_k)
        assert topk_ids.dtype in (torch.int32, torch.int64)

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, top_k),
            device=router_logits.device,
            dtype=router_logits.dtype,
        )
    else:
        assert topk_weights.is_cuda and topk_weights.shape == (num_tokens, top_k)
        assert topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)

    block_e = triton.next_power_of_2(num_experts)
    block_topk = triton.next_power_of_2(top_k)
    num_warps = 4
    if block_e >= 2048:
        num_warps = 8

    _moe_select_topk_softmax_kernel[(num_tokens,)](
        router_logits,
        topk_ids,
        topk_weights,
        num_experts,
        top_k,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_E=block_e,
        BLOCK_TOPK=block_topk,
        num_warps=num_warps,
    )

    return topk_ids, topk_weights


def moe_select_topk_softmax_with_counts(
    router_logits: torch.Tensor,
    top_k: int,
    num_experts: int | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
):
    """
    Select top-k experts and count selected tokens per expert in one Triton kernel.

    counts is zeroed by this function before launch.
    """
    assert router_logits.is_cuda
    assert router_logits.dim() == 2
    assert router_logits.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_tokens, router_num_experts = router_logits.shape
    if num_experts is None:
        num_experts = router_num_experts
    assert num_experts == router_num_experts
    assert 1 <= top_k <= num_experts

    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, top_k),
            device=router_logits.device,
            dtype=torch.int64,
        )
    else:
        assert topk_ids.is_cuda and topk_ids.shape == (num_tokens, top_k)
        assert topk_ids.dtype in (torch.int32, torch.int64)

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, top_k),
            device=router_logits.device,
            dtype=router_logits.dtype,
        )
    else:
        assert topk_weights.is_cuda and topk_weights.shape == (num_tokens, top_k)
        assert topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if counts is None:
        counts = torch.empty((num_experts,), device=router_logits.device, dtype=torch.int64)
    else:
        assert counts.is_cuda and counts.shape == (num_experts,)
        assert counts.dtype == torch.int64

    _zero_int64_triton(counts)

    block_e = triton.next_power_of_2(num_experts)
    block_topk = triton.next_power_of_2(top_k)
    num_warps = 4
    if block_e >= 2048:
        num_warps = 8

    _moe_select_topk_softmax_counts_kernel[(num_tokens,)](
        router_logits,
        topk_ids,
        topk_weights,
        counts,
        num_experts,
        top_k,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_E=block_e,
        BLOCK_TOPK=block_topk,
        num_warps=num_warps,
    )

    return topk_ids, topk_weights, counts


@triton.jit
def _moe_count_experts_kernel(
    topk_ids_ptr,
    counts_ptr,
    T: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_im,
    stride_ik,
    BLOCK_T: tl.constexpr,
):
    expert_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_T)
    total = tl.full((), 0, dtype=tl.int64)

    for start in tl.range(0, T, BLOCK_T):
        flat = start + offs
        mask = flat < T
        token_ids = flat // TOP_K
        route_ids = flat - token_ids * TOP_K
        route_experts = tl.load(
            topk_ids_ptr + token_ids * stride_im + route_ids * stride_ik,
            mask=mask,
            other=-1,
        )
        total += tl.sum(tl.where(route_experts == expert_id, 1, 0), axis=0)

    tl.store(counts_ptr + expert_id, total)


def moe_count_experts(
    topk_ids: torch.Tensor,
    num_experts: int,
    counts: torch.Tensor | None = None,
):
    """
    Count selected routes per expert using Triton instead of torch.bincount.
    """
    assert topk_ids.is_cuda
    assert topk_ids.dim() == 2
    assert topk_ids.dtype in (torch.int32, torch.int64)

    num_tokens, top_k = topk_ids.shape
    T = num_tokens * top_k

    if counts is None:
        counts = torch.empty((num_experts,), device=topk_ids.device, dtype=torch.int64)
    else:
        assert counts.is_cuda and counts.shape == (num_experts,)
        assert counts.dtype == torch.int64

    block_t = 1024
    _moe_count_experts_kernel[(num_experts,)](
        topk_ids,
        counts,
        T,
        top_k,
        topk_ids.stride(0),
        topk_ids.stride(1),
        BLOCK_T=block_t,
        num_warps=4,
    )
    return counts


@triton.jit
def _moe_prefix_counts_offsets_cursor_kernel(
    counts_ptr,
    expert_offsets_ptr,
    expert_cursor_ptr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS

    counts = tl.load(counts_ptr + offs, mask=mask, other=0).to(tl.int64)
    inclusive = tl.cumsum(counts, axis=0)
    exclusive = inclusive - counts

    tl.store(expert_offsets_ptr + offs, exclusive, mask=mask)
    tl.store(expert_cursor_ptr + offs, tl.zeros((BLOCK_E,), dtype=tl.int64), mask=mask)

    total = tl.sum(counts, axis=0)
    tl.store(expert_offsets_ptr + NUM_EXPERTS, total)


@triton.jit
def _moe_assign_sorted_positions_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    expert_offsets_ptr,
    expert_cursor_ptr,
    sorted_token_ids_ptr,
    sorted_token_weights_ptr,
    T: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < T

    token_ids = offs // TOP_K
    k_ids = offs - token_ids * TOP_K

    expert_ids = tl.load(
        topk_ids_ptr + token_ids * TOP_K + k_ids,
        mask=mask,
        other=0,
    )

    weights = tl.load(
        topk_weights_ptr + token_ids * TOP_K + k_ids,
        mask=mask,
        other=0.0,
    )

    local_pos = tl.atomic_add(
        expert_cursor_ptr + expert_ids,
        1,
        mask=mask,
        sem="relaxed",
    )

    expert_start = tl.load(
        expert_offsets_ptr + expert_ids,
        mask=mask,
        other=0,
    )

    sorted_pos = expert_start + local_pos

    tl.store(sorted_token_ids_ptr + sorted_pos, token_ids, mask=mask)
    tl.store(sorted_token_weights_ptr + sorted_pos, weights, mask=mask)


def build_moe_dispatch_metadata_fast(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    sorted_token_ids: torch.Tensor | None = None,
    sorted_token_weights: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_cursor: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
):
    """
    Build MoE dispatch metadata without materializing x_sorted.

    Returns:
        sorted_token_ids:     [T]
        sorted_token_weights: [T]
        expert_offsets:       [E + 1]
        counts:               [E]
    """
    assert topk_ids.is_cuda and topk_weights.is_cuda
    assert topk_ids.dim() == 2 and topk_weights.dim() == 2
    assert topk_ids.shape == topk_weights.shape
    assert topk_ids.dtype in (torch.int32, torch.int64)
    assert topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_tokens, top_k = topk_ids.shape
    T = num_tokens * top_k
    device = topk_ids.device

    if counts is None:
        counts = moe_count_experts(topk_ids, num_experts)
    else:
        assert counts.shape == (num_experts,)
        assert counts.dtype == torch.int64 and counts.is_cuda

    if expert_offsets is None:
        expert_offsets = torch.empty((num_experts + 1,), device=device, dtype=torch.int64)
    else:
        assert expert_offsets.shape == (num_experts + 1,)
        assert expert_offsets.dtype == torch.int64 and expert_offsets.is_cuda

    if expert_cursor is None:
        expert_cursor = torch.empty((num_experts,), device=device, dtype=torch.int64)
    else:
        assert expert_cursor.shape == (num_experts,)
        assert expert_cursor.dtype == torch.int64 and expert_cursor.is_cuda

    block_e = triton.next_power_of_2(num_experts)
    _moe_prefix_counts_offsets_cursor_kernel[(1,)](
        counts,
        expert_offsets,
        expert_cursor,
        num_experts,
        BLOCK_E=block_e,
        num_warps=1,
    )

    if sorted_token_ids is None:
        sorted_token_ids = torch.empty((T,), device=device, dtype=torch.int64)
    else:
        assert sorted_token_ids.shape == (T,)
        assert sorted_token_ids.dtype in (torch.int32, torch.int64)

    if sorted_token_weights is None:
        sorted_token_weights = torch.empty((T,), device=device, dtype=topk_weights.dtype)
    else:
        assert sorted_token_weights.shape == (T,)
        assert sorted_token_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)

    block_size = 256
    grid = (triton.cdiv(T, block_size),)

    _moe_assign_sorted_positions_kernel[grid](
        topk_ids,
        topk_weights,
        expert_offsets,
        expert_cursor,
        sorted_token_ids,
        sorted_token_weights,
        T,
        top_k,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return sorted_token_ids, sorted_token_weights, expert_offsets, counts


def build_grouped_tile_offsets(
    expert_offsets: torch.Tensor,
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Lightweight grouped-GEMM tile metadata.

    Instead of materializing per-tile arrays:
        tile_expert_ids / tile_m_ids / tile_n_ids

    only build:
        expert_tile_offsets: [E + 1]

    Kernel then decodes expert_id / pid_m / pid_n from tile_id.
    """
    assert expert_offsets.is_cuda and expert_offsets.dim() == 1

    counts = expert_offsets[1:] - expert_offsets[:-1]
    m_tiles = torch.div(counts + BLOCK_M - 1, BLOCK_M, rounding_mode="floor")
    n_tiles = triton.cdiv(N, BLOCK_N)
    tiles_per_expert = m_tiles * n_tiles

    expert_tile_offsets = torch.empty_like(expert_offsets)
    expert_tile_offsets[0] = 0
    expert_tile_offsets[1:] = torch.cumsum(tiles_per_expert, dim=0)

    # grid needs a Python int. This sync is much cheaper than building O(total_tiles) metadata.
    total_tiles = int(expert_tile_offsets[-1].item())
    return expert_tile_offsets, total_tiles


@triton.jit
def _moe_build_tile_offsets_kernel(
    expert_offsets_ptr,
    expert_tile_offsets_ptr,
    NUM_EXPERTS: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS

    starts = tl.load(expert_offsets_ptr + offs, mask=mask, other=0)
    ends = tl.load(expert_offsets_ptr + offs + 1, mask=mask, other=0)
    counts = ends - starts
    m_tiles = tl.cdiv(counts, BLOCK_M)
    tiles = m_tiles * N_TILES

    inclusive = tl.cumsum(tiles, axis=0)
    exclusive = inclusive - tiles

    tl.store(expert_tile_offsets_ptr + offs, exclusive, mask=mask)
    total = tl.sum(tiles, axis=0)
    tl.store(expert_tile_offsets_ptr + NUM_EXPERTS, total)


@triton.jit
def _moe_build_tile_offsets_pair_kernel(
    expert_offsets_ptr,
    expert_tile_offsets_1_ptr,
    expert_tile_offsets_2_ptr,
    NUM_EXPERTS: tl.constexpr,
    N_TILES_1: tl.constexpr,
    N_TILES_2: tl.constexpr,
    BLOCK_M_1: tl.constexpr,
    BLOCK_M_2: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < NUM_EXPERTS

    starts = tl.load(expert_offsets_ptr + offs, mask=mask, other=0)
    ends = tl.load(expert_offsets_ptr + offs + 1, mask=mask, other=0)
    counts = ends - starts

    tiles_1 = tl.cdiv(counts, BLOCK_M_1) * N_TILES_1
    inclusive_1 = tl.cumsum(tiles_1, axis=0)
    exclusive_1 = inclusive_1 - tiles_1
    tl.store(expert_tile_offsets_1_ptr + offs, exclusive_1, mask=mask)
    tl.store(expert_tile_offsets_1_ptr + NUM_EXPERTS, tl.sum(tiles_1, axis=0))

    tiles_2 = tl.cdiv(counts, BLOCK_M_2) * N_TILES_2
    inclusive_2 = tl.cumsum(tiles_2, axis=0)
    exclusive_2 = inclusive_2 - tiles_2
    tl.store(expert_tile_offsets_2_ptr + offs, exclusive_2, mask=mask)
    tl.store(expert_tile_offsets_2_ptr + NUM_EXPERTS, tl.sum(tiles_2, axis=0))


@triton.jit
def _decode_expert_from_tile_binary(
    tile_id,
    expert_tile_offsets_ptr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_EXPERT_SEARCH: tl.constexpr,
):
    lo = tl.full((), 0, dtype=tl.int64)
    hi = tl.full((), NUM_EXPERTS, dtype=tl.int64)

    for _ in tl.static_range(0, BLOCK_EXPERT_SEARCH):
        mid = (lo + hi) // 2
        mid_next = tl.minimum(mid + 1, NUM_EXPERTS)
        end_mid = tl.load(expert_tile_offsets_ptr + mid_next)
        go_right = tile_id >= end_mid
        lo = tl.where(go_right, mid_next, lo)
        hi = tl.where(go_right, hi, mid)

    expert_id = lo
    expert_tile_start = tl.load(expert_tile_offsets_ptr + expert_id)
    return expert_id, expert_tile_start


@triton.jit
def _decode_expert_from_tile_linear(
    tile_id,
    expert_tile_offsets_ptr,
    NUM_EXPERTS: tl.constexpr,
):
    expert_id = tl.full((), 0, dtype=tl.int64)
    expert_tile_start = tl.full((), 0, dtype=tl.int64)

    for e in range(0, NUM_EXPERTS):
        start_e = tl.load(expert_tile_offsets_ptr + e)
        end_e = tl.load(expert_tile_offsets_ptr + e + 1)
        in_expert = (tile_id >= start_e) & (tile_id < end_e)
        expert_id = tl.where(in_expert, e, expert_id)
        expert_tile_start = tl.where(in_expert, start_e, expert_tile_start)

    return expert_id, expert_tile_start


def build_grouped_tile_offsets_no_sync(
    expert_offsets: torch.Tensor,
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
    total_tokens: int,
    persistent_waves: int = 3,
):
    """
    Build grouped-GEMM tile offsets without synchronizing on total_tiles.

    The returned tile count is the number of persistent programs to launch.
    Kernels read expert_tile_offsets[-1] on device and loop over real tiles.
    """
    assert expert_offsets.is_cuda and expert_offsets.dim() == 1

    n_tiles = triton.cdiv(N, BLOCK_N)
    expert_tile_offsets = torch.empty_like(expert_offsets)

    num_experts = expert_offsets.numel() - 1
    block_e = triton.next_power_of_2(num_experts)
    _moe_build_tile_offsets_kernel[(1,)](
        expert_offsets,
        expert_tile_offsets,
        num_experts,
        n_tiles,
        BLOCK_M,
        BLOCK_E=block_e,
        num_warps=1,
    )

    launch_tiles = _persistent_launch_tiles(
        expert_offsets,
        total_tokens,
        N,
        BLOCK_M,
        BLOCK_N,
        persistent_waves,
    )
    return expert_tile_offsets, launch_tiles


def _persistent_launch_tiles(
    expert_offsets: torch.Tensor,
    total_tokens: int,
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
    persistent_waves: int,
):
    num_experts = expert_offsets.numel() - 1
    n_tiles = triton.cdiv(N, BLOCK_N)
    active_experts_upper_bound = min(num_experts, total_tokens)
    m_tiles_upper_bound = active_experts_upper_bound
    if total_tokens > active_experts_upper_bound:
        m_tiles_upper_bound += (total_tokens - active_experts_upper_bound) // BLOCK_M
    max_tiles = m_tiles_upper_bound * n_tiles

    sm_count = torch.cuda.get_device_properties(expert_offsets.device).multi_processor_count
    return min(max_tiles, max(1, sm_count * persistent_waves))


def build_grouped_tile_offsets_pair_no_sync(
    expert_offsets: torch.Tensor,
    N1: int,
    N2: int,
    BLOCK_M1: int,
    BLOCK_N1: int,
    BLOCK_M2: int,
    BLOCK_N2: int,
    total_tokens: int,
    persistent_waves1: int = 3,
    persistent_waves2: int = 3,
):
    assert expert_offsets.is_cuda and expert_offsets.dim() == 1

    num_experts = expert_offsets.numel() - 1
    block_e = triton.next_power_of_2(num_experts)
    n_tiles_1 = triton.cdiv(N1, BLOCK_N1)
    n_tiles_2 = triton.cdiv(N2, BLOCK_N2)

    expert_tile_offsets_1 = torch.empty_like(expert_offsets)
    expert_tile_offsets_2 = torch.empty_like(expert_offsets)

    _moe_build_tile_offsets_pair_kernel[(1,)](
        expert_offsets,
        expert_tile_offsets_1,
        expert_tile_offsets_2,
        num_experts,
        n_tiles_1,
        n_tiles_2,
        BLOCK_M1,
        BLOCK_M2,
        BLOCK_E=block_e,
        num_warps=1,
    )

    launch_tiles_1 = _persistent_launch_tiles(
        expert_offsets,
        total_tokens,
        N1,
        BLOCK_M1,
        BLOCK_N1,
        persistent_waves1,
    )
    launch_tiles_2 = _persistent_launch_tiles(
        expert_offsets,
        total_tokens,
        N2,
        BLOCK_M2,
        BLOCK_N2,
        persistent_waves2,
    )

    return (expert_tile_offsets_1, launch_tiles_1), (expert_tile_offsets_2, launch_tiles_2)


@triton.jit
def _moe_gemm1_silu_indirect_kernel(
    x_ptr,
    w13_ptr,
    sorted_token_ids_ptr,
    hidden_ptr,
    expert_offsets_ptr,
    expert_tile_offsets_ptr,
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
    GROUP_M: tl.constexpr,
    N_TILES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    LAUNCH_TILES: tl.constexpr,
    BLOCK_EXPERT_SEARCH: tl.constexpr,
    USE_BINARY_DECODE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = tl.load(expert_tile_offsets_ptr + NUM_EXPERTS)

    for tile_id in tl.range(pid, total_tiles, LAUNCH_TILES, flatten=True):
        if USE_BINARY_DECODE:
            expert_id, expert_tile_start = _decode_expert_from_tile_binary(
                tile_id,
                expert_tile_offsets_ptr,
                NUM_EXPERTS,
                BLOCK_EXPERT_SEARCH,
            )
        else:
            expert_id, expert_tile_start = _decode_expert_from_tile_linear(
                tile_id,
                expert_tile_offsets_ptr,
                NUM_EXPERTS,
            )

        local_tile = tile_id - expert_tile_start

        expert_start = tl.load(expert_offsets_ptr + expert_id)
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        expert_m = expert_end - expert_start
        m_tiles = tl.cdiv(expert_m, BLOCK_M)

        full_groups = m_tiles // GROUP_M
        full_group_tiles = full_groups * GROUP_M * N_TILES
        in_full_groups = local_tile < full_group_tiles

        group_id_full = local_tile // (GROUP_M * N_TILES)
        inside_full = local_tile - group_id_full * GROUP_M * N_TILES
        pid_n_full = inside_full // GROUP_M
        local_m_full = inside_full - pid_n_full * GROUP_M
        pid_m_full = group_id_full * GROUP_M + local_m_full

        rest_tile = local_tile - full_group_tiles
        last_group_size = m_tiles - full_groups * GROUP_M
        pid_n_last = rest_tile // last_group_size
        local_m_last = rest_tile - pid_n_last * last_group_size
        pid_m_last = full_groups * GROUP_M + local_m_last

        pid_m = tl.where(in_full_groups, pid_m_full, pid_m_last)
        pid_n = tl.where(in_full_groups, pid_n_full, pid_n_last)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        global_m = expert_start + offs_m

        token_ids = tl.load(
            sorted_token_ids_ptr + global_m,
            mask=offs_m < expert_m,
            other=0,
        )

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k

            x_ptrs = x_ptr + token_ids[:, None] * stride_xm + k_idxs[None, :] * stride_xk

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

        hidden_ptrs = hidden_ptr + global_m[:, None] * stride_hm + offs_n[None, :] * stride_hi
        out_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < I)
        tl.store(hidden_ptrs, hidden, mask=out_mask)


def moe_gemm1_silu_indirect(
    x: torch.Tensor,
    w13: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    hidden: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_metadata: tuple[torch.Tensor, int] | None = None,
    cfg: dict | None = None,
):
    assert x.is_cuda and w13.is_cuda and sorted_token_ids.is_cuda
    assert hidden.is_cuda and expert_offsets.is_cuda
    assert x.dim() == 2 and w13.dim() == 3 and hidden.dim() == 2

    _, H = x.shape
    E, H_w, two_I = w13.shape
    T, I = hidden.shape

    assert H == H_w and two_I == 2 * I
    assert sorted_token_ids.shape == (T,)
    assert expert_offsets.shape == (E + 1,)
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert w13.dtype == x.dtype and hidden.dtype == x.dtype
    assert w13.is_contiguous(), "w13 must be contiguous with shape [E, H, 2I]"

    cfg = DEFAULT_GEMM1_SILU_CFG if cfg is None else cfg

    if tile_metadata is None:
        tile_metadata = build_grouped_tile_offsets(
            expert_offsets,
            I,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
        )

    expert_tile_offsets, total_tiles = tile_metadata
    if total_tiles == 0:
        return hidden

    _moe_gemm1_silu_indirect_kernel[(total_tiles,)](
        x,
        w13,
        sorted_token_ids,
        hidden,
        expert_offsets,
        expert_tile_offsets,
        H,
        I,
        x.stride(0),
        x.stride(1),
        w13.stride(0),
        w13.stride(1),
        w13.stride(2),
        hidden.stride(0),
        hidden.stride(1),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        N_TILES=triton.cdiv(I, cfg["BLOCK_N"]),
        NUM_EXPERTS=E,
        LAUNCH_TILES=total_tiles,
        BLOCK_EXPERT_SEARCH=(E - 1).bit_length(),
        USE_BINARY_DECODE=E > LINEAR_EXPERT_DECODE_MAX_EXPERTS,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return hidden


@triton.jit
def _moe_gemm2_combine_kernel(
    hidden_ptr,
    w2_ptr,
    sorted_token_ids_ptr,
    sorted_token_weights_ptr,
    output_ptr,
    expert_offsets_ptr,
    expert_tile_offsets_ptr,
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
    GROUP_M: tl.constexpr,
    N_TILES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
    LAUNCH_TILES: tl.constexpr,
    BLOCK_EXPERT_SEARCH: tl.constexpr,
    USE_BINARY_DECODE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tiles = tl.load(expert_tile_offsets_ptr + NUM_EXPERTS)

    for tile_id in tl.range(pid, total_tiles, LAUNCH_TILES, flatten=True):
        if USE_BINARY_DECODE:
            expert_id, expert_tile_start = _decode_expert_from_tile_binary(
                tile_id,
                expert_tile_offsets_ptr,
                NUM_EXPERTS,
                BLOCK_EXPERT_SEARCH,
            )
        else:
            expert_id, expert_tile_start = _decode_expert_from_tile_linear(
                tile_id,
                expert_tile_offsets_ptr,
                NUM_EXPERTS,
            )

        local_tile = tile_id - expert_tile_start

        expert_start = tl.load(expert_offsets_ptr + expert_id)
        expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
        expert_m = expert_end - expert_start
        m_tiles = tl.cdiv(expert_m, BLOCK_M)

        full_groups = m_tiles // GROUP_M
        full_group_tiles = full_groups * GROUP_M * N_TILES
        in_full_groups = local_tile < full_group_tiles

        group_id_full = local_tile // (GROUP_M * N_TILES)
        inside_full = local_tile - group_id_full * GROUP_M * N_TILES
        pid_n_full = inside_full // GROUP_M
        local_m_full = inside_full - pid_n_full * GROUP_M
        pid_m_full = group_id_full * GROUP_M + local_m_full

        rest_tile = local_tile - full_group_tiles
        last_group_size = m_tiles - full_groups * GROUP_M
        pid_n_last = rest_tile // last_group_size
        local_m_last = rest_tile - pid_n_last * last_group_size
        pid_m_last = full_groups * GROUP_M + local_m_last

        pid_m = tl.where(in_full_groups, pid_m_full, pid_m_last)
        pid_n = tl.where(in_full_groups, pid_n_full, pid_n_last)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        global_m = expert_start + offs_m

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k

            hidden_ptrs = hidden_ptr + global_m[:, None] * stride_hm + k_idxs[None, :] * stride_hk
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
        output_ptrs = output_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < N)

        if USE_ATOMIC:
            tl.atomic_add(output_ptrs, vals, mask=out_mask, sem="relaxed")
        else:
            tl.store(output_ptrs, vals, mask=out_mask)


def moe_gemm2_combine(
    hidden: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_token_weights: torch.Tensor,
    output: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_metadata: tuple[torch.Tensor, int] | None = None,
    cfg: dict | None = None,
    use_atomic: bool = True,
):
    assert hidden.is_cuda and w2.is_cuda and sorted_token_ids.is_cuda
    assert sorted_token_weights.is_cuda and output.is_cuda and expert_offsets.is_cuda
    assert hidden.dim() == 2 and w2.dim() == 3 and output.dim() == 2

    T, I = hidden.shape
    E, I_w, H = w2.shape
    num_tokens, H_out = output.shape

    assert I == I_w and H == H_out
    assert expert_offsets.shape == (E + 1,)
    assert sorted_token_ids.shape == (T,)
    assert sorted_token_weights.shape == (T,)
    assert hidden.dtype in (torch.float16, torch.bfloat16)
    assert w2.dtype == hidden.dtype and output.dtype == hidden.dtype
    assert w2.is_contiguous(), "w2 must be contiguous with shape [E, I, H]"

    cfg = DEFAULT_GEMM2_COMBINE_CFG if cfg is None else cfg

    if tile_metadata is None:
        tile_metadata = build_grouped_tile_offsets(
            expert_offsets,
            H,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
        )

    expert_tile_offsets, total_tiles = tile_metadata
    if total_tiles == 0:
        return output

    _moe_gemm2_combine_kernel[(total_tiles,)](
        hidden,
        w2,
        sorted_token_ids,
        sorted_token_weights,
        output,
        expert_offsets,
        expert_tile_offsets,
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
        GROUP_M=cfg["GROUP_M"],
        N_TILES=triton.cdiv(H, cfg["BLOCK_N"]),
        NUM_EXPERTS=E,
        USE_ATOMIC=use_atomic,
        LAUNCH_TILES=total_tiles,
        BLOCK_EXPERT_SEARCH=(E - 1).bit_length(),
        USE_BINARY_DECODE=E > LINEAR_EXPERT_DECODE_MAX_EXPERTS,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return output


def fused_moe_prepared(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_token_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    output: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    gemm1_cfg: dict | None = None,
    gemm2_cfg: dict | None = None,
    gemm1_tile_metadata: tuple[torch.Tensor, int] | None = None,
    gemm2_tile_metadata: tuple[torch.Tensor, int] | None = None,
    use_atomic: bool = True,
    clear_output: bool = True,
):
    """
    Kernel-only fused MoE path.

    Metadata is already prepared:
        sorted_token_ids / sorted_token_weights / expert_offsets

    Computation:
        1. GEMM1 + SiLU*Up, indirect load from original x
        2. GEMM2 + combine
    """
    assert x.is_cuda and w13.is_cuda and w2.is_cuda
    assert sorted_token_ids.is_cuda and sorted_token_weights.is_cuda and expert_offsets.is_cuda

    num_tokens, H = x.shape
    E, H_w13, two_I = w13.shape
    E_w2, I, H_w2 = w2.shape
    T = sorted_token_ids.numel()

    assert H == H_w13 and H == H_w2 and E == E_w2 and two_I == 2 * I
    assert expert_offsets.shape == (E + 1,)

    gemm1_cfg = DEFAULT_GEMM1_SILU_CFG if gemm1_cfg is None else gemm1_cfg
    gemm2_cfg = DEFAULT_GEMM2_COMBINE_CFG if gemm2_cfg is None else gemm2_cfg

    if output is None:
        output = torch.empty((num_tokens, H), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (num_tokens, H) and output.dtype == x.dtype

    if clear_output:
        _zero_tensor_triton(output)

    if hidden is None:
        hidden = torch.empty((T, I), device=x.device, dtype=x.dtype)
    else:
        assert hidden.shape == (T, I) and hidden.dtype == x.dtype

    if gemm1_tile_metadata is None:
        gemm1_tile_metadata = build_grouped_tile_offsets(
            expert_offsets,
            I,
            BLOCK_M=gemm1_cfg["BLOCK_M"],
            BLOCK_N=gemm1_cfg["BLOCK_N"],
        )

    if gemm2_tile_metadata is None:
        gemm2_tile_metadata = build_grouped_tile_offsets(
            expert_offsets,
            H,
            BLOCK_M=gemm2_cfg["BLOCK_M"],
            BLOCK_N=gemm2_cfg["BLOCK_N"],
        )

    moe_gemm1_silu_indirect(
        x,
        w13,
        sorted_token_ids,
        hidden,
        expert_offsets,
        tile_metadata=gemm1_tile_metadata,
        cfg=gemm1_cfg,
    )

    moe_gemm2_combine(
        hidden,
        w2,
        sorted_token_ids,
        sorted_token_weights,
        output,
        expert_offsets,
        tile_metadata=gemm2_tile_metadata,
        cfg=gemm2_cfg,
        use_atomic=use_atomic,
    )

    return output


def _cfg_key(cfg: dict):
    return (
        cfg["BLOCK_M"],
        cfg["BLOCK_N"],
        cfg["BLOCK_K"],
        cfg["GROUP_M"],
        cfg["num_warps"],
        cfg["num_stages"],
    )


def _time_cuda_callable(fn, repeats: int = 3):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _autotune_persistent_waves(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_token_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    gemm1_cfg: dict,
    gemm2_cfg: dict,
    top_k: int,
    use_atomic: bool,
    clear_output: bool,
    gemm1_candidates: tuple[int, ...] = DEFAULT_GEMM1_PERSISTENT_WAVES_CANDIDATES,
    gemm2_candidates: tuple[int, ...] = DEFAULT_GEMM2_PERSISTENT_WAVES_CANDIDATES,
):
    num_tokens, H = x.shape
    E, _, two_I = w13.shape
    I = two_I // 2
    T = sorted_token_ids.numel()
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    key = (
        device_index,
        str(x.dtype),
        num_tokens,
        E,
        H,
        I,
        top_k,
        _cfg_key(gemm1_cfg),
        _cfg_key(gemm2_cfg),
        use_atomic,
        clear_output,
    )
    cached = _PERSISTENT_WAVES_CACHE.get(key)
    if cached is not None:
        return cached

    tune_output = torch.empty((num_tokens, H), device=x.device, dtype=x.dtype)
    tune_hidden = torch.empty((T, I), device=x.device, dtype=x.dtype)

    best_waves = (gemm1_candidates[0], gemm2_candidates[0])
    best_ms = float("inf")

    for g1_waves in gemm1_candidates:
        for g2_waves in gemm2_candidates:
            g1_meta, g2_meta = build_grouped_tile_offsets_pair_no_sync(
                expert_offsets,
                I,
                H,
                BLOCK_M1=gemm1_cfg["BLOCK_M"],
                BLOCK_N1=gemm1_cfg["BLOCK_N"],
                BLOCK_M2=gemm2_cfg["BLOCK_M"],
                BLOCK_N2=gemm2_cfg["BLOCK_N"],
                total_tokens=T,
                persistent_waves1=g1_waves,
                persistent_waves2=g2_waves,
            )

            def run_candidate():
                fused_moe_prepared(
                    x,
                    w13,
                    w2,
                    sorted_token_ids,
                    sorted_token_weights,
                    expert_offsets,
                    output=tune_output,
                    hidden=tune_hidden,
                    gemm1_cfg=gemm1_cfg,
                    gemm2_cfg=gemm2_cfg,
                    gemm1_tile_metadata=g1_meta,
                    gemm2_tile_metadata=g2_meta,
                    use_atomic=use_atomic,
                    clear_output=clear_output,
                )

            run_candidate()
            elapsed_ms = _time_cuda_callable(run_candidate)
            if elapsed_ms < best_ms:
                best_ms = elapsed_ms
                best_waves = (g1_waves, g2_waves)

    _PERSISTENT_WAVES_CACHE[key] = best_waves
    return best_waves


def fused_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor | None = None,
    top_k: int | None = None,
    output: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    sorted_token_ids: torch.Tensor | None = None,
    sorted_token_weights: torch.Tensor | None = None,
    expert_offsets: torch.Tensor | None = None,
    expert_cursor: torch.Tensor | None = None,
    expert_counts: torch.Tensor | None = None,
    gemm1_cfg: dict | None = None,
    gemm2_cfg: dict | None = None,
    use_atomic: bool | None = None,
    clear_output: bool | None = None,
    persistent_waves: int | tuple[int, int] | str = "auto",
):
    """
    Public end-to-end fused MoE path.

    Inputs:
        x:            [num_tokens, H]
        topk_ids:     [num_tokens, top_k], optional when router_logits is provided
        topk_weights: [num_tokens, top_k], optional when router_logits is provided
        w13:          [E, H, 2I]
        w2:           [E, I, H]
        router_logits:[num_tokens, E], optional; used to select top-k routes in Triton
        top_k:        required when router_logits is provided and topk buffers are omitted
        expert_counts:[E], optional workspace used to skip torch.bincount on logits routing
        persistent_waves: "auto" autotunes no-sync GEMM launch waves, an int pins
            both GEMMs, and (g1, g2) pins them separately

    This path does not materialize x_sorted.
    """
    assert x.is_cuda and w13.is_cuda and w2.is_cuda
    assert x.dim() == 2

    num_tokens, H = x.shape
    E, H_w13, two_I = w13.shape
    E_w2, I, H_w2 = w2.shape

    assert H == H_w13 and H == H_w2 and E == E_w2 and two_I == 2 * I

    if router_logits is not None:
        assert router_logits.is_cuda and router_logits.dim() == 2
        assert router_logits.shape == (num_tokens, E)
        if top_k is None:
            if topk_ids is not None:
                top_k = topk_ids.shape[1]
            elif topk_weights is not None:
                top_k = topk_weights.shape[1]
            else:
                raise ValueError("top_k is required when routing from router_logits")
        topk_ids, topk_weights, expert_counts = moe_select_topk_softmax_with_counts(
            router_logits,
            top_k,
            num_experts=E,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            counts=expert_counts,
        )
    else:
        if topk_ids is None or topk_weights is None:
            raise ValueError("topk_ids/topk_weights or router_logits/top_k must be provided")
        assert topk_ids.is_cuda and topk_weights.is_cuda
        assert topk_ids.dim() == 2 and topk_weights.dim() == 2
        assert topk_ids.shape == topk_weights.shape
        top_k = topk_ids.shape[1]

    if use_atomic is None:
        use_atomic = top_k != 1
    if clear_output is None:
        clear_output = use_atomic

    gemm1_cfg = DEFAULT_GEMM1_SILU_CFG if gemm1_cfg is None else gemm1_cfg
    gemm2_cfg = DEFAULT_GEMM2_COMBINE_CFG if gemm2_cfg is None else gemm2_cfg

    sorted_token_ids, sorted_token_weights, expert_offsets, _ = build_moe_dispatch_metadata_fast(
        topk_ids,
        topk_weights,
        E,
        sorted_token_ids=sorted_token_ids,
        sorted_token_weights=sorted_token_weights,
        expert_offsets=expert_offsets,
        expert_cursor=expert_cursor,
        counts=expert_counts,
    )

    total_expanded_tokens = num_tokens * top_k

    if persistent_waves == "auto":
        persistent_waves = _autotune_persistent_waves(
            x,
            w13,
            w2,
            sorted_token_ids,
            sorted_token_weights,
            expert_offsets,
            gemm1_cfg,
            gemm2_cfg,
            top_k,
            use_atomic,
            clear_output,
        )
    elif isinstance(persistent_waves, int):
        assert persistent_waves >= 1
        persistent_waves = (persistent_waves, persistent_waves)
    else:
        assert isinstance(persistent_waves, tuple) and len(persistent_waves) == 2
        assert persistent_waves[0] >= 1 and persistent_waves[1] >= 1

    gemm1_persistent_waves, gemm2_persistent_waves = persistent_waves

    gemm1_tile_metadata, gemm2_tile_metadata = build_grouped_tile_offsets_pair_no_sync(
        expert_offsets,
        I,
        H,
        BLOCK_M1=gemm1_cfg["BLOCK_M"],
        BLOCK_N1=gemm1_cfg["BLOCK_N"],
        BLOCK_M2=gemm2_cfg["BLOCK_M"],
        BLOCK_N2=gemm2_cfg["BLOCK_N"],
        total_tokens=total_expanded_tokens,
        persistent_waves1=gemm1_persistent_waves,
        persistent_waves2=gemm2_persistent_waves,
    )

    return fused_moe_prepared(
        x,
        w13,
        w2,
        sorted_token_ids,
        sorted_token_weights,
        expert_offsets,
        output=output,
        hidden=hidden,
        gemm1_cfg=gemm1_cfg,
        gemm2_cfg=gemm2_cfg,
        gemm1_tile_metadata=gemm1_tile_metadata,
        gemm2_tile_metadata=gemm2_tile_metadata,
        use_atomic=use_atomic,
        clear_output=clear_output,
    )
