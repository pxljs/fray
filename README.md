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
