import torch

import fray
from fray import bench_kineto


RTOL = 1e-2
ATOL = 5e-1


def compare_outputs(out_triton: torch.Tensor, out_ref: torch.Tensor):
    torch.cuda.synchronize()

    diff = torch.abs(out_triton.float() - out_ref.float())

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat = diff.flatten()

    max_quantile_elements = 1_000_000

    if diff.numel() > max_quantile_elements:
        idx = torch.randint(0, diff.numel(), (max_quantile_elements,), device=diff.device)
        diff_sample = flat[idx]
    else:
        diff_sample = flat
    
    p99_diff = torch.quantile(diff_sample, 0.99).item()
    p999_diff = torch.quantile(diff_sample, 0.999).item()

    is_close = torch.allclose(
        out_triton,
        out_ref,
        rtol=RTOL,
        atol=ATOL,
    )

    return {
        "is_close": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "p99_diff": p99_diff,
        "p999_diff": p999_diff,
    }


def print_compare_result(stats):
    print(f"Accuracy Check      : {'✅ PASS' if stats['is_close'] else '❌ FAIL'}")
    print(f"Max Diff            : {stats['max_diff']:.6f}")
    print(f"Mean Diff           : {stats['mean_diff']:.6f}")
    print(f"P99 Diff            : {stats['p99_diff']:.6f}")
    print(f"P999 Diff           : {stats['p999_diff']:.6f}")


def matmul_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton MatMul vs PyTorch torch.mm")
    print("=" * 60)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (16, 16, 16),
        (32, 64, 32),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (1024, 4096, 1024),
        (1024, 11008, 4096),
        (4096, 4096, 1024),
    ]

    for M, K, N in test_cases:
        print(f"\nShape: A=[{M}, {K}], B=[{K}, {N}], C=[{M}, {N}], dtype={dtype}")

        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        out_triton = torch.empty((M, N), device=device, dtype=dtype)

        fray.triton.matmul(a, b, out_triton)
        out_ref = torch.mm(a, b)

        stats = compare_outputs(out_triton, out_ref)

        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print_compare_result(stats)

    print("=" * 60 + "\n")


def test_matmul_non_multiple_shapes():
    print("\n" + "=" * 60)
    print("Non-multiple Shape Test")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (127, 255, 129),
        (513, 1025, 257),
        (1000, 4097, 1001),
        (2049, 4096, 513),
        (1025, 11009, 4097),
    ]

    for M, K, N in test_cases:
        print(f"\nShape: A=[{M}, {K}], B=[{K}, {N}], C=[{M}, {N}], dtype={dtype}")

        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        out_triton = torch.empty((M, N), device=device, dtype=dtype)
        out_ref = torch.empty((M, N), device=device, dtype=dtype)

        fray.triton.matmul(a, b, out_triton)
        torch.mm(a, b, out=out_ref)

        stats = compare_outputs(out_triton, out_ref)

        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print_compare_result(stats)

    print("=" * 60 + "\n")


def test_matmul_strided_inputs():
    print("\n" + "=" * 60)
    print("Strided Input Test")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (128, 256, 128),
        (512, 1024, 512),
        (1024, 4096, 1024),
        (1024, 11008, 4096),
    ]

    for M, K, N in test_cases:
        print(f"\nShape: A=[{M}, {K}], B=[{K}, {N}], C=[{M}, {N}], dtype={dtype}")
        print("A and B are non-contiguous views")

        # a_base: [K, M] -> a: [M, K], stride 非 contiguous
        a_base = torch.randn((K, M), device=device, dtype=dtype)
        a = a_base.t()

        # b_base: [N, K] -> b: [K, N], stride 非 contiguous
        b_base = torch.randn((N, K), device=device, dtype=dtype)
        b = b_base.t()

        assert not a.is_contiguous()
        assert not b.is_contiguous()
        assert a.shape == (M, K)
        assert b.shape == (K, N)

        out_triton = torch.empty((M, N), device=device, dtype=dtype)
        out_ref = torch.empty((M, N), device=device, dtype=dtype)

        fray.triton.matmul(a, b, out_triton)
        torch.mm(a, b, out=out_ref)

        stats = compare_outputs(out_triton, out_ref)

        print(f"A stride            : {a.stride()}")
        print(f"B stride            : {b.stride()}")
        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print_compare_result(stats)

    print("=" * 60 + "\n")


def test_matmul_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton MatMul vs PyTorch torch.mm")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        # square GEMM
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),

        # LLM-like projection shapes
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),

        # FFN / MoE-like shapes
        (1024, 4096, 11008),
        (2048, 4096, 11008),
        (4096, 4096, 11008),
        (1024, 11008, 4096),
        (2048, 11008, 4096),
        (4096, 11008, 4096),
    ]

    for M, K, N in test_cases:
        print(f"\nConfiguration: M={M}, K={K}, N={N}, dtype={dtype}")

        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        out_triton = torch.empty((M, N), device=device, dtype=dtype)
        out_torch = torch.empty((M, N), device=device, dtype=dtype)

        def run_triton():
            fray.triton.matmul(a, b, out_triton)

        def run_torch():
            torch.mm(a, b, out=out_torch)

        # 显式触发 Triton JIT / cuBLAS 初始化，避免污染正式计时
        run_triton()
        run_torch()
        torch.cuda.synchronize()

        stats = compare_outputs(out_triton, out_torch)

        if not stats["is_close"]:
            print(
                f"WARNING: correctness mismatch, "
                f"max_diff={stats['max_diff']:.6f}, "
                f"mean_diff={stats['mean_diff']:.6f}"
            )

        t_torch_s = bench_kineto(run_torch, "torch_matmul")
        t_triton_s = bench_kineto(run_triton, "triton_matmul")

        total_flops = 2.0 * M * N * K

        triton_tflops = total_flops / t_triton_s / 1e12
        torch_tflops = total_flops / t_torch_s / 1e12

        total_bytes = (
            M * K * a.element_size()
            + K * N * b.element_size()
            + M * N * out_triton.element_size()
        )

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(
            f"Triton MatMul  : {t_triton_s * 1e6:8.2f} us | "
            f"Compute: {triton_tflops:8.2f} TFLOPS | "
            f"BW: {triton_gbps:8.2f} GB/s"
        )
        print(
            f"PyTorch torch.mm: {t_torch_s * 1e6:8.2f} us | "
            f"Compute: {torch_tflops:8.2f} TFLOPS | "
            f"BW: {torch_gbps:8.2f} GB/s"
        )

        if t_triton_s > 0:
            print(f"Speedup        : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff       : {stats['max_diff']:.6f}")
        print(f"Mean Diff      : {stats['mean_diff']:.6f}")
        print(f"P99 Diff       : {stats['p99_diff']:.6f}")
        print(f"P999 Diff      : {stats['p999_diff']:.6f}")
        print("-" * 60)


def test_matmul_strided_performance():
    print("\n" + "=" * 60)
    print("Strided Input Performance Benchmark")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (512, 1024, 512),
        (1024, 4096, 1024),
        (1024, 11008, 4096),
    ]

    for M, K, N in test_cases:
        print(f"\nConfiguration: M={M}, K={K}, N={N}, dtype={dtype}")
        print("A and B are non-contiguous views")

        a_base = torch.randn((K, M), device=device, dtype=dtype)
        a = a_base.t()

        b_base = torch.randn((N, K), device=device, dtype=dtype)
        b = b_base.t()

        out_triton = torch.empty((M, N), device=device, dtype=dtype)
        out_torch = torch.empty((M, N), device=device, dtype=dtype)

        def run_triton():
            fray.triton.matmul(a, b, out_triton)

        def run_torch():
            torch.mm(a, b, out=out_torch)

        run_triton()
        run_torch()
        torch.cuda.synchronize()

        stats = compare_outputs(out_triton, out_torch)

        if not stats["is_close"]:
            print(
                f"WARNING: correctness mismatch, "
                f"max_diff={stats['max_diff']:.6f}, "
                f"mean_diff={stats['mean_diff']:.6f}"
            )

        t_torch_s = bench_kineto(run_torch, "torch_strided_matmul")
        t_triton_s = bench_kineto(run_triton, "triton_strided_matmul")

        total_flops = 2.0 * M * N * K

        triton_tflops = total_flops / t_triton_s / 1e12
        torch_tflops = total_flops / t_torch_s / 1e12

        print("-" * 60)
        print(f"A stride       : {a.stride()}")
        print(f"B stride       : {b.stride()}")
        print(f"Triton MatMul  : {t_triton_s * 1e6:8.2f} us | Compute: {triton_tflops:8.2f} TFLOPS")
        print(f"PyTorch torch.mm: {t_torch_s * 1e6:8.2f} us | Compute: {torch_tflops:8.2f} TFLOPS")

        if t_triton_s > 0:
            print(f"Speedup        : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff       : {stats['max_diff']:.6f}")
        print(f"Mean Diff      : {stats['mean_diff']:.6f}")
        print("-" * 60)


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    matmul_accuracy_test()

    print("\nStep 2: Running Non-multiple Shape Test...")
    test_matmul_non_multiple_shapes()

    print("\nStep 3: Running Strided Input Test...")
    test_matmul_strided_inputs()

    print("\nStep 4: Running Performance Test...")
    test_matmul_performance()

    print("\nStep 5: Running Strided Input Performance Test...")
    test_matmul_strided_performance()