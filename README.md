⚡ Fray
========
A JIT-based High-Performance Operator Framework for Deep Learning

📖 Overview
-----------

Fray is a Just-In-Time (JIT) compiled, high-performance CUDA operator framework designed for
modern Deep Learning systems. Built on top of NVIDIA's **CuTe** (the core algebraic layout engine
of CUTLASS 3.x) and integrated seamlessly with PyTorch's C++ Extension JIT compiler, Fray provides
a dynamic, auto-tunable, and zero-overhead execution environment for heavily optimized GPU kernels.

Fray bridges the gap between high-level Python models and low-level hardware micro-architecture,
allowing for dynamic shape tuning, aggressive register reuse, and instruction-level latency hiding
on NVIDIA Ampere (SM80) and later architectures.

🏗 Architecture
---------------

```
Python DSL (Fray Kernels)
    │
    ├── JIT Compiler ──── Generates CUDA code from C++ templates
    │       │
    │       ├── CuTe (CUTLASS Algebraic Layout Engine)
    │       ├── CUTLASS (Tiled MMA, Copy, Epilogue)
    │       └── Fray Headers (Custom kernel implementations)
    │
    ├── Auto-Tuner ────── Compiles & profiles multiple tile configurations
    │       │
    │       ├── L2 cache flush between runs
    │       ├── Iterative profiling (20 warmup runs)
    │       └── Best config caching (persisted to disk)
    │
    └── Runtime ────────── Dynamically loads compiled .so/.dll via ctypes
```

The framework follows a three-stage pipeline:

1. **Compile** — Python kernel definitions are instantiated as C++ templates with concrete tile
   sizes, compiled on-the-fly via NVCC (≥12.3), and cached to disk for reuse across sessions.
2. **Tune** — Multiple tile configurations are benchmarked with L2 cache flushing to find the
   optimal block sizes for the current GPU architecture.
3. **Invoke** — The best kernel is loaded as a shared library and called directly from Python
   with zero framework overhead.

✨ Key Features
---------------

- **JIT Compilation with Caching** — Kernels are compiled on first use and cached by content hash.
  Subsequent runs skip compilation entirely.
- **Auto-Tuning** — Each kernel explores a configurable search space of tile sizes, automatically
  selecting the fastest variant for the current GPU.
- **CuTe-Native** — All kernels are built directly on CuTe primitives (TiledCopy, TiledMMA,
  Swizzle layouts), bypassing CUTLASS composition layers for maximum control over the
  instruction stream.
- **FFMA Interleave SASS Optimization** — Post-compilation FFMA (Fused Floating-point
  Multiply-Add) instruction interleaving to hide instruction latency on SM80/SM89/SM90.
- **Non-Contiguous Tensor Support** — CuTe's algebraic layout system enables kernels to handle
  arbitrary strided tensors without intermediate copies.
- **Cross-Platform** — Supports both Linux and Windows with automatic NVCC detection.

📦 Supported Kernels
--------------------

| Kernel | Precision | Description |
|--------|-----------|-------------|
| `flashattn_cute` | FP16 | Flash Attention with online softmax fusion, CuTe-native implementation |
| `fp16_gemm` | FP16 → FP32 accumulate | Tiled GEMM with asynchronous shared memory pipelining |
| `online_softmax` | FP32 | Numerically stable online softmax with vectorized float4 loads |
| `softmax` | FP32 | Multi-dimensional softmax supporting arbitrary strided layouts |
| `reduce_sum_max` | FP32 | Fused sum + max reduction with float4 vectorized access |

🚀 Quick Start
--------------

### Prerequisites

- CUDA Toolkit ≥ 12.3
- PyTorch ≥ 2.0 with CUDA support
- NVIDIA GPU with compute capability ≥ SM80 (Ampere, Ada, Hopper)

### Installation

```bash
pip install -e .
```

### Usage

```python
import torch
import fray

# FP16 GEMM — C = A × Bᵀ
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, dtype=torch.half, device='cuda')
b = torch.randn(N, K, dtype=torch.half, device='cuda')
c = torch.zeros(M, N, dtype=torch.half, device='cuda')
fray.jit_kernels.fp16_gemm(a, b, c)

# Flash Attention
B, H, S, D = 2, 16, 2048, 128
q = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
k = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
v = torch.randn(B, H, S, D, dtype=torch.half, device='cuda')
out = torch.zeros_like(q)
fray.jit_kernels.flash_attn_cute(q, k, v, out)

# Online Softmax (FP32)
x = torch.randn(16384, 512, dtype=torch.float, device='cuda')
y = torch.empty_like(x)
fray.jit_kernels.online_softmax(x, y)

# Fused Reduce (Sum + Max)
x = torch.randn(4096 * 1024, dtype=torch.float, device='cuda')
y_sum = torch.zeros(1, dtype=torch.float, device='cuda')
y_max = torch.full((1,), float('-inf'), dtype=torch.float, device='cuda')
fray.jit_kernels.reduce_sum_max(x, y_sum, y_max)
```

### Running Tests

```bash
python tests/test_fp16_gemm.py
python tests/test_flashattn_cute.py
python tests/test_online_softmax.py
python tests/test_softmax.py
python tests/test_reduce.py
```

### Running Benchmarks

```python
from fray import bench_kineto
import fray

def my_kernel():
    fray.jit_kernels.fp16_gemm(a, b, c)

avg_time_s = bench_kineto(my_kernel, 'fray_cute_gemm')
print(f"Average time: {avg_time_s * 1e6:.2f} us")
```

🔧 Configuration
----------------

| Environment Variable | Description |
|----------------------|-------------|
| `FRAY_CACHE_DIR` | Override the default cache directory (`~/.cache/fray`) |
| `FRAY_NVCC_COMPILER` | Path to a specific NVCC binary |
| `FRAY_JIT_DEBUG` | Print generated CUDA code and compilation commands |
| `FRAY_PRINT_AUTOTUNE` | Print auto-tuning results |
| `FRAY_PTXAS_VERBOSE` | Enable PTX assembler verbose output (register usage) |
| `FRAY_DISABLE_FFMA_INTERLEAVE` | Disable FFMA interleaving SASS optimization |

📁 Project Structure
--------------------

```
fray/
├── fray/
│   ├── __init__.py
│   ├── utils.py                  # bench_kineto, calc_diff utilities
│   ├── jit/
│   │   ├── __init__.py
│   │   ├── compiler.py           # NVCC detection, JIT build pipeline
│   │   ├── runtime.py            # ctypes-based dynamic library loader
│   │   ├── template.py           # C++ code generation from templates
│   │   └── interleave_ffma.py    # SASS-level FFMA interleaving optimizer
│   ├── jit_kernels/
│   │   ├── __init__.py
│   │   ├── tuner.py              # Auto-tuner with profiling & caching
│   │   ├── flashattn_cute.py     # Flash Attention kernel definition
│   │   ├── fp16_gemm.py          # FP16 GEMM kernel definition
│   │   ├── online_softmax.py     # Online Softmax kernel definition
│   │   ├── softmax.py            # Multi-dimensional Softmax kernel definition
│   │   └── reduce.py             # Fused Reduce kernel definition
│   └── include/
│       ├── flash_attn/
│       │   ├── flashattn_cute.cuh
│       │   └── softmax.cuh
│       ├── gemm/
│       │   └── fp16_gemm.cuh
│       ├── softmax/
│       │   ├── softmax.cuh
│       │   └── online_softmax.cuh
│       └── reduce/
│           └── reduce.cuh
├── tests/
│   ├── test_flashattn_cute.py
│   ├── test_fp16_gemm.py
│   ├── test_online_softmax.py
│   ├── test_softmax.py
│   ├── test_reduce.py
│   └── test_jit.py
├── third-party/
│   ├── cutlass/                  # CUTLASS + CuTe headers
│   ├── flashinfer/               # FlashInfer reference headers
│   ├── ThunderKittens/           # ThunderKittens reference
│   └── xqa/                      # XQA kernel reference implementations
├── pyproject.toml
├── setup.py
└── README.md
```

📊 Benchmarks
-------------

Performance comparisons against PyTorch native backends (cuBLAS for GEMM, SDPA for attention)
on RTX 4090 (Ada Lovelace, SM89).

### FP16 GEMM

| Shape (M×N×K) | Fray (TFLOPS) | cuBLAS (TFLOPS) | Speedup |
|---------------|---------------|-----------------|---------|
| 4096×5120×5120 | — | — | — |
| 4096×1536×24576 | — | — | — |
| 4096×16384×7168 | — | — | — |

### Flash Attention

| Config (B×H×S×D) | Fray (TFLOPS) | SDPA (TFLOPS) | Speedup |
|-------------------|---------------|---------------|---------|
| 8×16×1024×64 | — | — | — |
| 8×64×2048×128 | — | — | — |
| 8×64×4096×128 | — | — | — |

> Run `python tests/test_fp16_gemm.py` and `python tests/test_flashattn_cute.py` to generate
> up-to-date numbers for your hardware.

📄 License
----------

[MIT](LICENSE)
