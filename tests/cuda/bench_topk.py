"""PYTHONPATH=. .venv/bin/python tests/cuda/bench_topk.py --help"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from fray.jit_kernels.topk import topk, _select_algorithm


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
CASES = [(1, 128, 8), (128, 128, 8), (4096, 128, 8), (1024, 1024, 16),
         (128, 4096, 8), (16, 16384, 32), (128, 1024, 512),
         (32, 4096, 2048), (16, 8193, 4097), (16, 16384, 12000)]


def elapsed(fn, *, mode, iterations, repeats):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        if mode == "graph":
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
        else:
            # Synchronized wall time includes the Python wrapper, allocation
            # when requested, launch overhead, and device execution.
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start_time) * 1e6 / iterations)
    return statistics.median(samples)


def validate(x, actual, expected):
    values, indices = actual
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(values, x.gather(-1, indices), rtol=0, atol=0)
    if indices.shape[-1] > 1:
        assert (indices.sort().values.diff() > 0).all(), "duplicate indices"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["auto", "reduce", "radix", "torch", "all"], default="auto")
    parser.add_argument("--dtype", choices=[*DTYPES, "all"], default="all")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("ROWS", "N", "K"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smallest", action="store_true")
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--allocate", action="store_true", help="include output allocation (eager only)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    if args.allocate and args.mode != "eager":
        parser.error("--allocate requires --mode eager")
    cases = [tuple(args.shape)] if args.shape else CASES
    if any(rows <= 0 or n <= 0 or not 0 < k <= n for rows, n, k in cases):
        parser.error("shape requires ROWS > 0 and 0 < K <= N")
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable; run this benchmark on the target GPU.\n")
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    print(torch.cuda.get_device_name(), "torch", torch.__version__, "CUDA", torch.version.cuda)
    print(f"mode={args.mode}, allocate={args.allocate}, largest={not args.smallest}; "
          "median warm timing, JIT excluded")
    print("dtype rows n k path | Fray us | torch us | speedup")
    records = []
    for name in DTYPES if args.dtype == "all" else [args.dtype]:
        for rows, n, k in cases:
            x = torch.randn(rows, n, device="cuda", dtype=DTYPES[name])
            a = (torch.empty(rows, k, device=x.device, dtype=x.dtype),
                 torch.empty(rows, k, device=x.device, dtype=torch.int64))
            b = tuple(torch.empty_like(t) for t in a)
            algorithms = ["auto", "reduce", "radix"] if args.algorithm == "all" else [args.algorithm]
            for algorithm in algorithms:
                if algorithm == "reduce" and (k > 32 or n > 16384):
                    continue
                if algorithm == "radix" and n > 16384:
                    continue
                selected = _select_algorithm(n, k) if algorithm == "auto" else algorithm
                tiled = selected == "radix" and n > 4096
                workspace_bytes = rows * ((n + 4095) // 4096) * 4096 * 8 if tiled else 0
                workspace = (torch.empty(workspace_bytes, device=x.device, dtype=torch.uint8)
                             if tiled and not args.allocate else None)
                def custom():
                    return topk(x, k, largest=not args.smallest, out=None if args.allocate else a,
                                algorithm=algorithm, workspace=workspace)
                def reference():
                    return torch.topk(x, k, largest=not args.smallest, out=None if args.allocate else b)
                validate(x, custom(), reference())
                options = dict(mode=args.mode, iterations=args.iterations, repeats=args.repeats)
                us, ref_us = elapsed(custom, **options), elapsed(reference, **options)
                path = "tiled-radix" if tiled else selected
                record = dict(dtype=name, rows=rows, n=n, k=k, algorithm=algorithm, path=path,
                              workspace_bytes=workspace_bytes, mode=args.mode, allocate=args.allocate,
                              largest=not args.smallest, fray_us=us, torch_us=ref_us, speedup=ref_us / us)
                records.append(record)
                print(name, rows, n, k, algorithm, path, f"| {us:.3f} | {ref_us:.3f} | {ref_us/us:.2f}x")
    if args.csv:
        if not records:
            parser.error("no supported cases for the selected algorithm")
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
