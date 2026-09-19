"""Ampere/Ada Top-K: register selection, block radix sort and tiled merge."""
import torch
from .tuner import jit_tuner


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Conservative byte-span check, including strided fallback inputs."""
    if a.device != b.device or not a.numel() or not b.numel():
        return False
    def end(t):
        span = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
        return t.data_ptr() + span * t.element_size()
    return a.data_ptr() < end(b) and b.data_ptr() < end(a)


def _validate_out(x, out, shape):
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError("out must be a (values, indices) pair")
    values, indices = out
    for name, t, dtype in (("values", values, x.dtype), ("indices", indices, torch.int64)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"out {name} must be a Tensor")
        if (t.layout != torch.strided or tuple(t.shape) != shape or t.dtype != dtype
                or t.device != x.device or not t.is_contiguous()):
            raise ValueError(f"out {name} must have matching shape/device/dtype and be contiguous")
    tensors = (x, values, indices)
    if any(_overlaps(a, b) for i, a in enumerate(tensors) for b in tensors[i + 1:]):
        raise ValueError("input and outputs must not overlap")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("topk with out does not support automatic differentiation")
    return values, indices


def _select_algorithm(n: int, k: int) -> str:
    """Initial crossover heuristic, not a measured hardware tuning table.

    Repeated selection grows with K. Block radix sorting has fixed pass count.
    Tiled radix needs a second launch, so retain selection longer for wide rows.
    """
    if n <= 0 or k == 0:
        return "reduce"
    if n > 16384:
        return "torch"
    threshold = 8 if n <= 1024 or n > 4096 else 4
    # Dense selections on tiny rows also benefit from sorting the row once.
    return "reduce" if k <= threshold and (k == 1 or k * 4 <= n) else "radix"


def topk(x: torch.Tensor, k: int, largest: bool = True, sorted: bool = True,
         *, dim: int = -1, out=None, algorithm: str = "auto", workspace=None):
    """Return top-k values and int64 indices (also accessible as .values/.indices).

    Custom paths select the last dimension of contiguous CUDA FP16/BF16/FP32
    inputs with width <= 16384, for every k. Auto dispatch uses register
    selection for small k, block radix sort for larger k, and tiled radix/merge
    for wide rows. Thresholds are provisional, pending target-GPU benchmarks.
    algorithm="reduce"/"radix" forces a custom path (invalid inputs raise);
    algorithm="torch" forces the reference. Reduce supports k <= 32.
    Other dimensions/layouts/dtypes and gradient-enabled inputs fall back in auto.
    Custom ties choose the lower index; fallback ties follow PyTorch.
    sorted=False may still return sorted results.

    Tiled radix optionally accepts a contiguous CUDA uint8 workspace with at
    least rows * ceil(width / 4096) * 4096 * 8 bytes. It must not overlap inputs
    or outputs. It is allocated when omitted. Do not share it between concurrent
    calls on different streams. Non-tiled paths reject supplied workspace.

    out=(values, indices) requires exact shapes, matching device/dtypes,
    contiguous buffers and disjoint byte spans, on both custom and fallback
    paths. Buffers are never resized. Compile/warm up before graph capture.
    Existing positional largest/sorted arguments are preserved; dim is keyword-only.
    """
    if not isinstance(algorithm, str) or algorithm not in ("auto", "reduce", "radix", "torch"):
        raise ValueError("algorithm must be auto, reduce, radix or torch")
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided:
        raise ValueError("topk expects a strided tensor")
    if x.ndim == 0:
        raise ValueError("topk expects at least one dimension")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if not -x.ndim <= dim < x.ndim:
        raise IndexError("dim is out of range")
    dim %= x.ndim
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 0 <= k <= x.shape[dim]:
        raise ValueError("k must be in [0, x.shape[dim]]")
    if not isinstance(largest, bool) or not isinstance(sorted, bool):
        raise TypeError("largest and sorted must be bool")
    shape = tuple(k if i == dim else size for i, size in enumerate(x.shape))
    if out is not None:
        out = _validate_out(x, out, shape)
    n = x.shape[-1]
    rows = x.numel() // n if n else 0
    custom = (x.is_cuda and x.is_contiguous() and dim == x.ndim - 1 and
              not (torch.is_grad_enabled() and x.requires_grad) and
              x.dtype in (torch.float16, torch.bfloat16, torch.float32) and
              n <= 16384 and rows * max(1, (n + 4095) // 4096) <= 2147483647)
    selected = (_select_algorithm(n, k) if custom else "torch") if algorithm == "auto" else algorithm
    if algorithm in ("reduce", "radix") and not custom:
        raise ValueError("forced custom algorithm requires contiguous CUDA FP16/BF16/FP32 last-dim inference input, width <= 16384")
    if selected == "reduce" and k > 32:
        raise ValueError("forced reduce supports k <= 32")
    custom = custom and selected != "torch"
    if not custom:
        if workspace is not None:
            raise ValueError("workspace is only used by tiled radix")
        return torch.topk(x, k, dim=dim, largest=largest, sorted=sorted, out=out)
    if out is None:
        values = torch.empty(shape, device=x.device, dtype=x.dtype)
        indices = torch.empty(shape, device=x.device, dtype=torch.int64)
    else:
        values, indices = out
    tiled = selected == "radix" and n > 4096 and rows > 0 and k > 0
    if workspace is not None:
        if not tiled:
            raise ValueError("workspace is only used by tiled radix")
        required = rows * ((n + 4095) // 4096) * 4096 * 8
        if (not isinstance(workspace, torch.Tensor) or workspace.dtype != torch.uint8
                or workspace.device != x.device or workspace.layout != torch.strided
                or not workspace.is_contiguous() or workspace.numel() < required
                or workspace.data_ptr() % 8):
            raise ValueError("workspace must be aligned contiguous CUDA uint8 storage of sufficient size")
        if any(_overlaps(workspace, t) for t in (x, values, indices)):
            raise ValueError("workspace must not overlap input or outputs")
    elif tiled:
        workspace = torch.empty(rows * ((n + 4095) // 4096) * 4096 * 8,
                                dtype=torch.uint8, device=x.device)
    if not rows or not k:
        return torch.return_types.topk((values, indices))
    threads = 32 if n <= 1024 else 256
    items = 1 << ((n + threads - 1) // threads - 1).bit_length()
    if selected == "radix":
        threads = 256
        items = min(16, 1 << ((n + 255) // 256 - 1).bit_length())
    ctype = {torch.float16: "__half", torch.bfloat16: "__nv_bfloat16",
             torch.float32: "float"}[x.dtype]
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream(x.device)
        # JIT's uint8 pointer ABI also carries the int64 output without changing
        # the shared JIT type registry or allocating an index conversion buffer.
        args = (x, values, indices.view(torch.uint8), rows, n, k, stream)
        arg_defs = (("X", x.dtype), ("V", x.dtype), ("I", torch.uint8),
                    ("rows", int), ("n", int), ("k", int), ("stream", torch.cuda.Stream))
        if selected == "radix":
            # Small rows do not dereference W; reuse an existing aligned pointer.
            args += (workspace if tiled else indices.view(torch.uint8),)
            arg_defs += (("W", torch.uint8),)
            template = "__return_code = static_cast<int>(fray::topk_radix_c<{T}, {ITEMS}, {LARGEST}>(X, V, reinterpret_cast<int64_t*>(I), reinterpret_cast<uint64_t*>(W), rows, n, k, stream));"
        else:
            template = "__return_code = static_cast<int>(fray::topk_c<{T}, {ITEMS}, {THREADS}, {LARGEST}>(X, V, reinterpret_cast<int64_t*>(I), rows, n, k, stream));"
        runtime = jit_tuner.compile_and_tune(
            name="topk_" + selected,
            keys={"T": ctype, "ITEMS": items, "THREADS": threads,
                  "LARGEST": "true" if largest else "false", "DEVICE": x.device.index},
            space=(), includes=('"topk/topk.cuh"',), arg_defs=arg_defs,
            template=template, args=args)
        code = runtime(*args)
        if code != 0:
            raise RuntimeError(f"topk CUDA launch failed with error {code}")
    return torch.return_types.topk((values, indices))
