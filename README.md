# Fray

A playground and toolkit for writing high-performance deep learning operators
with CUDA/CuTe JIT kernels and Triton kernels.

Fray is focused on the parts of GPU programming that matter when building modern
LLM inference kernels: explicit memory movement, tiled GEMM, online reductions,
attention variants, routing, grouped GEMM, and fused MoE execution. The project
contains two complementary implementation paths:

- `fray.jit_kernels`: CUDA/CuTe kernels compiled just in time and cached.
- `fray.triton`: Triton kernels for fast iteration and end-to-end operator
  prototyping.

## Highlights

- JIT-compiled CUDA/CuTe kernels with content-hash based caching.
- Triton implementations for common operators and MoE workflows.
- Auto-tuning support for CUDA JIT kernels.
- Focused tests and benchmarks under `tests/cuda` and `tests/triton`.
- Reference-oriented third-party sources under `third-party`.

## Supported Operators

| Area | API | Backend | Notes |
| --- | --- | --- | --- |
| Vector add | `fray.triton.vector_add` | Triton | Minimal Triton example |
| Matmul | `fray.triton.matmul` | Triton | FP16/BF16 style tiled matmul |
| Grouped GEMM | `fray.triton.grouped_gemm` | Triton | Metadata-driven grouped GEMM |
| RMSNorm | `fray.triton.rmsnorm`, `fray.triton.add_rmsnorm`, `fray.jit_kernels.fused_rmsnorm` | Triton, CUDA/CuTe | Normalization kernels |
| RoPE | `fray.triton.rope`, `fray.jit_kernels.fused_rope` | Triton, CUDA/CuTe | GPT-NeoX style rotary embedding |
| SiLU and multiply | `fray.triton.silu_mul` | Triton | MoE activation helper |
| GELU and multiply | `fray.triton.gelu_mul` | Triton | GeGLU activation helper |
| Softmax | `fray.triton.softmax`, `fray.jit_kernels.softmax` | Triton, CUDA/CuTe | Dense softmax kernels |
| Online softmax | `fray.jit_kernels.online_softmax` | CUDA/CuTe | Streaming softmax reduction |
| FP16 GEMM | `fray.jit_kernels.fp16_gemm` | CUDA/CuTe | Tiled GEMM |
| Flash decoding | `fray.jit_kernels.flash_decoding` | CUDA/CuTe | Decode attention path |
| Flash MLA | `fray.jit_kernels.flash_mla` | CUDA/CuTe | MLA-oriented attention kernel |
| FlashAttention | `fray.jit_kernels.flash_attn_cute` | CUDA/CuTe | CuTe-native attention experiment |
| Fused MoE | `fray.triton.fused_moe` | Triton | Routing, dispatch metadata, two GEMMs, combine |

## Installation

Fray requires Python 3.12+, PyTorch with CUDA, Triton, and a CUDA toolchain for
the CUDA/CuTe JIT kernels.

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[test,bench,dev]"
```

The project also includes `uv.lock`, so `uv` can be used if you prefer a locked
environment workflow.

## Quick Start

### Triton Matmul

```python
import torch
from fray.triton import matmul

m, n, k = 4096, 4096, 4096
a = torch.randn((m, k), device="cuda", dtype=torch.float16)
b = torch.randn((k, n), device="cuda", dtype=torch.float16)

out = matmul(a, b)
```

### Triton Fused MoE

```python
import torch
from fray.triton import fused_moe

num_tokens = 4096
num_experts = 64
hidden_size = 4096
intermediate_size = 14336
top_k = 2

x = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.float16)
router_logits = torch.randn(
    (num_tokens, num_experts), device="cuda", dtype=torch.float32
)
w13 = torch.randn(
    (num_experts, hidden_size, 2 * intermediate_size),
    device="cuda",
    dtype=torch.float16,
)
w2 = torch.randn(
    (num_experts, intermediate_size, hidden_size),
    device="cuda",
    dtype=torch.float16,
)

out = fused_moe(x, router_logits, w13, w2, top_k=top_k)
```

### CUDA/CuTe JIT GEMM

```python
import torch
import fray

m, n, k = 4096, 4096, 4096
a = torch.randn((m, k), dtype=torch.float16, device="cuda")
b = torch.randn((n, k), dtype=torch.float16, device="cuda")
c = torch.empty((m, n), dtype=torch.float16, device="cuda")

fray.jit_kernels.fp16_gemm(a, b, c)
```

## Tests

CUDA/CuTe JIT tests:

```bash
pytest tests/cuda
```

Triton tests:

```bash
pytest tests/triton
```

Run a focused MoE test or benchmark:

```bash
pytest tests/triton/test_fused_moe.py
```

Some tests require a CUDA GPU and may compile kernels on first run.

## Benchmarking

Use `fray.bench_kineto` for timing small callables:

```python
from fray import bench_kineto

avg_time_s = bench_kineto(lambda: fused_moe(x, router_logits, w13, w2, top_k=2),
                          "fused_moe")
print(f"{avg_time_s * 1e6:.2f} us")
```

The fused MoE tests include prepared and end-to-end benchmark paths. Prepared
benchmarks measure the core compute path with dispatch metadata supplied.
End-to-end benchmarks include routing and dispatch metadata construction.

## Configuration

| Environment variable | Description |
| --- | --- |
| `FRAY_CACHE_DIR` | Override the CUDA JIT cache directory. |
| `FRAY_NVCC_COMPILER` | Select a specific `nvcc` binary. |
| `FRAY_JIT_DEBUG` | Print generated CUDA code and build commands. |
| `FRAYJIT_PRINT_NVCC_COMMAND` | Print only the NVCC build command. |
| `FRAY_JIT_MAX_WORKERS` | Limit parallel NVCC compilations during tuning. |
| `FRAY_PRINT_AUTOTUNE` | Print auto-tuning results. |
| `FRAY_PTXAS_VERBOSE` | Enable ptxas verbose output. |
| `FRAY_DISABLE_FFMA_INTERLEAVE` | Disable FFMA interleaving optimization. |

## Project Structure

```text
fray/
├── fray/
│   ├── __init__.py
│   ├── _version.py
│   ├── utils.py
│   ├── jit/
│   │   ├── compiler.py
│   │   ├── runtime.py
│   │   ├── template.py
│   │   └── interleave_ffma.py
│   ├── jit_kernels/
│   │   ├── flash_decoding.py
│   │   ├── flash_mla.py
│   │   ├── flashattn_cute.py
│   │   ├── fp16_gemm.py
│   │   ├── online_softmax.py
│   │   ├── reduce.py
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   ├── softmax.py
│   │   └── tuner.py
│   ├── triton/
│   │   ├── fused_moe.py
│   │   ├── gelu_mul.py
│   │   ├── grouped_gemm.py
│   │   ├── matmul.py
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   ├── silu_mul.py
│   │   ├── softmax.py
│   │   └── vector_add.py
│   └── include/
│       ├── flash_attn/
│       ├── flash_mla/
│       ├── fused_moe/
│       ├── gemm/
│       ├── norm/
│       ├── reduce/
│       ├── rope/
│       └── softmax/
├── tests/
│   ├── cuda/
│   └── triton/
├── third-party/
│   ├── cutlass/
│   ├── flashinfer/
│   ├── ThunderKittens/
│   └── xqa/
├── pyproject.toml
├── setup.py
├── uv.lock
└── README.md
```

### Module Roles

- `fray/jit`: generic CUDA JIT infrastructure.
- `fray/jit_kernels`: Python-facing CUDA/CuTe kernel wrappers.
- `fray/include`: CUDA headers and CuTe kernel implementations.
- `fray/triton`: Triton operator implementations and public Triton APIs.
- `tests/cuda`: correctness and smoke tests for CUDA/CuTe kernels.
- `tests/triton`: correctness, diagnostics, and benchmark-oriented Triton tests.
- `third-party`: vendored or reference implementations used while developing
  kernels.

## Development Notes

- Keep CUDA/CuTe JIT code and Triton code separated unless a shared utility is
  genuinely backend-agnostic.
- Put public Triton entry points in `fray/triton/__init__.py`.
- Add focused tests beside the backend being changed: `tests/cuda` for JIT CUDA
  kernels and `tests/triton` for Triton kernels.
- Prefer prepared benchmark paths when measuring kernel compute time, and
  end-to-end benchmark paths when measuring real operator latency.

## License

License file not included yet.

### Triton Top-K and routed grouped GEMM

```python
from fray.triton import topk, routed_grouped_gemm
selected = topk(logits, 16, algorithm="auto")
output = routed_grouped_gemm(x, expert_weights, logits, top_k=4, combine="auto")
# x[M,H], expert_weights[E,H,N], logits[M,E] -> output[M,N]
```

Custom Top-K handles contiguous CUDA FP16/BF16/FP32 last-axis inputs with width
<= 16384 and every legal K. Auto keeps repeated reduction for small K, adds
`partial` for width > 1024 and 8 < K <= 64, and otherwise sorts. Partial sorts
256-element tiles, keeps next_power_of_2(K) candidates per tile, and sorts only
that candidate pool. Local top-K suffices because an excluded value already has
at least K better elements in its own tile. Thresholds are provisional; force
`reduce` (K <= 32), `partial` (1 <= K <= 64), `sort`, or `torch` to benchmark.
Full sorting uses 4096-element tiles for wide rows and a parallel rank merge.
Only ceil(min(K,4096)/256) merge programs launch per tile. Searches only visit
other tiles' first min(K,4096) positions and mask candidates already ranked >= K.

FP16/BF16 comparisons pack native 16-bit ordered values and 16-bit indices into
32-bit keys. FP32 uses 64-bit keys. Equal values prefer lower indices; signed
zeros compare equal and NaNs sort first for largest, last for smallest. Returned
values are read from original input to preserve their bits. Outputs use int64
indices. Custom sorted=False is allowed to remain sorted. Auto falls back to
PyTorch for unsupported inputs/autograd; forced custom algorithms reject them.

`workspace=` optionally reuses contiguous same-device torch.uint32 (FP16/BF16)
or torch.uint64 (FP32) scratch storage, disjoint from input/output. Required
number of elements is rows * ceil(width/256) * next_power_of_2(K) for partial,
or rows * ceil(width/4096) * 4096 for tiled sort. Do not reuse concurrently
across streams. Without workspace the wrapper allocates it. Other custom paths
do not use scratch. Warm up before graph capture.

The shared selection primitive remains fused with selected-logit softmax and
expert counts in the MoE routing kernels for up to 4096 experts. Wider expert
rows use standalone selection plus a normalization/counting kernel. All counts
are reset per invocation. Nonfinite logits retain ordinary softmax NaN behavior.

`routed_grouped_gemm` now uses GPU-built tile offsets and persistent scheduling,
with indirect input loads from sorted token IDs. It does not materialize an
expanded input or call the legacy CPU tile-list builder. GEMM keeps FP16/BF16
inputs and FP32 accumulation. Its epilogue rounds the projection to input dtype,
multiplies the routing weight, and rounds the weighted contribution to input
dtype before combining (matching the previous per-route rounding boundary).
`combine="atomic"` writes contributions into a zeroed FP32 output, while
`"staged"` writes weighted route outputs, then gathers them by token using a
dispatch-produced inverse route map and sums in fixed route order with FP32
accumulation. It writes the final output dtype directly, without output zeroing,
FP32 route-buffer conversion or scatter atomics. `"scatter"` retains the old
FP32 index_add path as an explicit benchmark baseline.
Both finally cast to input dtype; accumulation can differ slightly from the
old repeated half-precision accumulation. Auto provisionally uses atomic for
K <= 4. Warmed calls support CUDA graph capture; intermediate allocation remains.
The old explicit-metadata `grouped_gemm` API remains available; its legacy
metadata helper still performs CPU readback. Its dot operands now retain their
input dtype rather than unconditionally being converted to FP32.

```sh
.venv/bin/python -m pytest tests/triton/test_topk.py -q
PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py
PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py --grouped --combine atomic
PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py --grouped --combine staged
PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py --grouped --combine scatter
PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py --breakdown
```

Standalone timing uses preallocated outputs/scratch and warm CUDA graphs.
Grouped timing compares complete eager wall time with independent PyTorch
routing, expert mm and weighted combination. Breakdown separately reports
routing/counts, dispatch, GPU tile metadata, GEMM/weights, and combination/cast
with prepared inputs. Atomic outputs are cleared every benchmark invocation;
these isolated device timings do not sum to eager end-to-end latency.
SM86/SM89 representative kernels compile offline, including FP16 MMA with FP32
accumulation. GPU correctness and speedups remain unverified until target-card
execution; no performance claim is inferred from compilation.

The staged inverse map costs `tokens * top_k * 8` bytes and is written during
the existing dispatch launch. The gather kernel uses one program per token and
128 output columns, sequentially accumulating route contributions; it trades
scatter contention for indexed reads. Which combine wins depends on shape and
cache behavior and still requires target-GPU measurement. Optional
`build_moe_dispatch_metadata_fast(..., route_positions=buffer)` fills a
contiguous int64 `[tokens * top_k]` map from flattened token/rank to sorted row;
the existing four return values are unchanged. The buffer must have independent
storage from other inputs/outputs. Atomic mode does not allocate this map.
