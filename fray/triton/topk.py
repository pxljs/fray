"""Triton Top-K with a shared, fusible row-selection primitive."""
import torch
import triton
import triton.language as tl


@triton.jit
def _comparison_keys(values, cols, N: tl.constexpr, LARGEST: tl.constexpr):
    if values.dtype == tl.float16 or values.dtype == tl.bfloat16:
        bits = tl.where(values == 0, 0, values).to(values.dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
        ordered = tl.where((bits & 0x8000) != 0, bits ^ 0xffff, bits ^ 0x8000)
        ordered = tl.where(values != values, 0xffff, ordered)
        if LARGEST:
            ordered = ordered ^ 0xffff
        keys = (ordered.to(tl.uint32) << 16) | cols.to(tl.uint32)
        return tl.where(cols < N, keys, 0xffffffff).to(tl.uint32)
    else:
        v = values.to(tl.float32)
        bits = tl.where(v == 0, 0., v).to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        ordered = tl.where(v != v, 0xffffffff, ordered).to(tl.uint32)
        if LARGEST:
            ordered = ~ordered
        keys = (ordered.to(tl.uint64) << 16) | cols.to(tl.uint64)
        return tl.where(cols < N, keys, 0xffffffffffffffff).to(tl.uint64)


@triton.jit
def select_topk_row(values, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    LARGEST: tl.constexpr, SORT: tl.constexpr):
    """Return FP32 values and int32 indices; padded output lanes are invalid."""
    cols = tl.arange(0, BLOCK_N)
    ranks = tl.arange(0, BLOCK_K)
    v = values.to(tl.float32)
    keys = _comparison_keys(values, cols, N, LARGEST)
    sentinel = ~tl.full((), 0, keys.dtype)
    if SORT:
        keys = tl.sort(keys, descending=False)
        chosen = tl.gather(keys, ranks, axis=0)
    else:
        chosen = ~tl.full((BLOCK_K,), 0, keys.dtype)
        for rank in tl.static_range(K):
            best = tl.min(keys, axis=0)
            chosen = tl.where(ranks == rank, best, chosen)
            keys = tl.where(keys == best, sentinel, keys)
    ids = (chosen & 0xffff).to(tl.int32)
    safe_ids = tl.where(ranks < K, ids, 0)
    selected = tl.gather(v, safe_ids, axis=0)
    return tl.where(ranks < K, selected, -float('inf')), ids


def _select_algorithm(n, k):
    # Bitonic sorting is O(N log^2 N); reduction is O(N*K).
    # Provisional crossover, not target-GPU measurements.
    return "reduce" if k <= 8 and (k == 1 or k * 4 <= n) else "sort"


@triton.jit
def _topk_kernel(X, V, Indices, N: tl.constexpr, K: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 LARGEST: tl.constexpr, SORT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    values = tl.load(X + row * N + cols, cols < N, other=0)
    _, ids = select_topk_row(values, N, K, BLOCK_N, BLOCK_K, LARGEST, SORT)
    ranks = tl.arange(0, BLOCK_K)
    # Reload original dtype to preserve signed zero and NaN payloads.
    result = tl.load(X + row * N + ids, ranks < K, other=0)
    tl.store(V + row * K + ranks, result, ranks < K)
    tl.store(Indices + row * K + ranks, ids, ranks < K)


@triton.jit
def _sort_tiles(X, Workspace, N: tl.constexpr, TILES: tl.constexpr, LARGEST: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * 4096 + tl.arange(0, 4096)
    v = tl.load(X + row * N + cols, cols < N, other=0)
    keys = _comparison_keys(v, cols, N, LARGEST)
    keys = tl.sort(keys, descending=False)
    tl.store(Workspace + (row * TILES + tile) * 4096 + tl.arange(0, 4096), keys)


@triton.jit
def _merge_tiles(X, V, Indices, Workspace, N: tl.constexpr, K: tl.constexpr,
                 TILES: tl.constexpr, LARGEST: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    programs: tl.constexpr = triton.cdiv(min(K, 4096), 256)
    tile = tl.program_id(1) // programs
    local = (tl.program_id(1) % programs) * 256 + tl.arange(0, 256)
    # Runs are ascending by the requested value order, then original index.
    keys = tl.load(Workspace + (row * TILES + tile) * 4096 + local)
    sentinel = ~tl.full((), 0, keys.dtype)
    valid = (keys != sentinel) & (local < K)
    rank = local
    for other in tl.static_range(TILES):
        lo = tl.full((256,), 0, tl.int32)
        hi = tl.full((256,), min(K, 4096, N - other * 4096), tl.int32)
        for _ in range(13):
            active = valid & (rank < K) & (other != tile) & (lo < hi)
            mid = (lo + hi) // 2
            candidate = tl.load(Workspace + (row * TILES + other) * 4096 + mid,
                                active, other=0)
            left = candidate < keys
            lo = tl.where(active & left, mid + 1, lo)
            hi = tl.where(active & ~left, mid, hi)
        rank += tl.where(other != tile, lo, 0)
    ids = (keys & 0xffff).to(tl.int32)
    mask = valid & (rank < K)
    value = tl.load(X + row * N + ids, mask, other=0)
    tl.store(V + row * K + rank, value, mask)
    tl.store(Indices + row * K + rank, ids, mask)


@triton.jit
def _candidate_tiles(X, Workspace, N: tl.constexpr, LARGEST: tl.constexpr,
                     TILES: tl.constexpr, C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * 256 + tl.arange(0, 256)
    v = tl.load(X + row * N + cols, cols < N, other=0)
    keys = tl.sort(_comparison_keys(v, cols, N, LARGEST), descending=False)
    candidates = tl.gather(keys, tl.arange(0, C), axis=0)
    tl.store(Workspace + (row * TILES + tile) * C + tl.arange(0, C), candidates)


@triton.jit
def _candidate_merge(X, V, Indices, Workspace, N: tl.constexpr, K: tl.constexpr,
                     COUNT: tl.constexpr, B: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, B)
    keys = tl.load(Workspace + row * COUNT + offsets, offsets < COUNT, other=-1)
    keys = tl.sort(keys, descending=False)
    selected = tl.gather(keys, tl.arange(0, BK), axis=0)
    ids = (selected & 0xffff).to(tl.int32)
    rank = tl.arange(0, BK)
    vals = tl.load(X + row * N + ids, rank < K, other=0)
    tl.store(V + row * K + rank, vals, rank < K)
    tl.store(Indices + row * K + rank, ids, rank < K)


def _standalone_algorithm(n, k):
    if 1024 < n <= 16384 and 8 < k <= 64:
        return "partial"
    return _select_algorithm(n, k)


def topk(x, k, largest=True, sorted=True, *, dim=-1, out=None, algorithm="auto", workspace=None):
    """Last-axis CUDA FP16/BF16/FP32 Top-K up to width 16384, all K.

    Auto uses reduction for small K, local candidates for intermediate K,
    and bitonic sorting for large K. Optional workspace reuses tiled scratch.
    Forced reduce (K <= 32), partial (K <= 64) / sort reject unsupported inputs. Auto/torch
    preserve autograd via torch.topk fallback. Custom ties favor smaller indices.
    sorted=False may still be sorted. Exact contiguous nonoverlapping out
    buffers are required. Warm up before CUDA graph capture.
    """
    if algorithm not in ("auto", "reduce", "sort", "partial", "torch"):
        raise ValueError("algorithm must be auto, reduce, sort, partial or torch")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided or x.ndim == 0:
        raise ValueError("x must be a non-scalar strided tensor")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if not -x.ndim <= dim < x.ndim:
        raise IndexError("dim out of range")
    dim %= x.ndim
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 0 <= k <= x.shape[dim]:
        raise ValueError("k out of range")
    if not isinstance(largest, bool) or not isinstance(sorted, bool):
        raise TypeError("largest and sorted must be bool")
    shape = tuple(k if d == dim else size for d, size in enumerate(x.shape))
    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise TypeError("out must be a values/indices pair")
        for t, dtype in zip(out, (x.dtype, torch.int64)):
            if not isinstance(t, torch.Tensor):
                raise TypeError("outputs must be tensors")
            if (t.layout != torch.strided or t.shape != shape or t.device != x.device
                    or t.dtype != dtype or not t.is_contiguous()):
                raise ValueError("invalid output shape/device/dtype/layout")
        tensors = (x, *out)
        spans = [(t.data_ptr(), t.data_ptr() + (1 + sum((s - 1) * st for s, st in
                  zip(t.shape, t.stride()))) * t.element_size()) if t.numel() else (0, 0)
                 for t in tensors]
        if any(a < d and c < b for j, (a, b) in enumerate(spans) for c, d in spans[j + 1:]):
            raise ValueError("input and output buffers overlap")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise RuntimeError("out does not support autograd")
        out = tuple(out)
    n = x.shape[-1]
    eligible = (x.is_cuda and x.is_contiguous() and dim == x.ndim - 1
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and n <= 16384 and (x.numel() // max(n, 1)) <= 2147483647
                and not (torch.is_grad_enabled() and x.requires_grad))
    if algorithm in ("reduce", "sort", "partial") and not eligible:
        raise ValueError("forced custom path requires contiguous CUDA last-axis inference input, width <= 16384")
    if algorithm == "reduce" and k > 32:
        raise ValueError("reduce requires k <= 32")
    if algorithm == "partial" and not 1 <= k <= 64:
        raise ValueError("partial requires 1 <= k <= 64")
    if not eligible or algorithm == "torch":
        if workspace is not None:
            raise ValueError("workspace requires a custom scratch path")
        return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted, out=out)
    if out is None:
        out = (torch.empty(shape, device=x.device, dtype=x.dtype),
               torch.empty(shape, device=x.device, dtype=torch.int64))
    if k and x.numel():
        selected = _standalone_algorithm(n, k) if algorithm == "auto" else algorithm
        rows = x.numel() // n
        scratch = selected == "partial" or (selected == "sort" and n > 4096)
        count = (triton.cdiv(n, 256) * triton.next_power_of_2(k) if selected == "partial"
                 else triton.cdiv(n, 4096) * 4096)
        dtype = torch.uint32 if x.dtype != torch.float32 else torch.uint64
        if workspace is not None:
            if not isinstance(workspace, torch.Tensor):
                raise TypeError("workspace must be a Tensor")
            if (not scratch or workspace.device != x.device or workspace.dtype != dtype
                    or not workspace.is_contiguous() or workspace.numel() < rows * count):
                raise ValueError("invalid workspace device/dtype/size/layout")
            for t in (x, *out):
                if (workspace.data_ptr() < t.data_ptr() + t.numel() * t.element_size()
                        and t.data_ptr() < workspace.data_ptr() + workspace.numel() * workspace.element_size()):
                    raise ValueError("workspace overlaps input/output")
        elif scratch:
            workspace = torch.empty((rows * count,), device=x.device, dtype=dtype)
        with torch.cuda.device(x.device):
            if selected == "partial":
                tiles = triton.cdiv(n, 256)
                candidates = triton.next_power_of_2(k)
                _candidate_tiles[(rows, tiles)](x, workspace, n, largest, tiles, candidates, num_warps=4)
                _candidate_merge[(rows,)](x, *out, workspace, n, k, count,
                                          triton.next_power_of_2(count), candidates, num_warps=8)
            elif selected == "sort" and n > 4096:
                tiles = triton.cdiv(n, 4096)
                _sort_tiles[(rows, tiles)](x, workspace, n, tiles, largest, num_warps=8)
                programs = triton.cdiv(min(k, 4096), 256)
                _merge_tiles[(rows, tiles * programs)](x, *out, workspace, n, k, tiles, largest, num_warps=4)
            else:
                _topk_kernel[(rows,)](
                    x, *out, n, k, triton.next_power_of_2(n), triton.next_power_of_2(k),
                    largest, selected == "sort", num_warps=4 if n < 2048 else 8)
    return torch.return_types.topk(out)
