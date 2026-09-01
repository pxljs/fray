import torch
import torch.nn.functional as F

import fray
from fray import bench_kineto


def torch_gelu_mul(gate: torch.Tensor, up: torch.Tensor):
    return F.gelu(gate) * up


def gelu_mul_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton GELU-Mul vs PyTorch F.gelu(x) * y")
    print("=" * 60)

    test_cases = [
        (1024, torch.float16),
        (4096, torch.float16),
        (1024 * 1024, torch.float16),
        (1024 * 1024 + 123, torch.float16),
        (1024 * 1024 + 31, torch.bfloat16),
    ]

    for n_elements, dtype in test_cases:
        gate = torch.randn(n_elements, device="cuda", dtype=dtype)
        up = torch.randn(n_elements, device="cuda", dtype=dtype)
        out_triton = torch.empty_like(gate)

        fray.triton.gelu_mul(gate, up, out_triton)
        out_ref = torch_gelu_mul(gate, up)
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()
        is_close = torch.allclose(out_triton, out_ref, rtol=1e-2, atol=1e-2)

        print(f"\nN = {n_elements}, dtype={dtype}")
        print(f"Triton Output Sample: {out_triton[:4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[:4].tolist()}")
        print(f"Accuracy Check      : {'PASS' if is_close else 'FAIL'}")
        print(f"Max Diff            : {max_diff:.6f}")
        print(f"Mean Diff           : {mean_diff:.6f}")

        assert is_close

    print("=" * 60 + "\n")


def test_gelu_mul_accuracy():
    gelu_mul_accuracy_test()


def test_gelu_mul_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton GELU-Mul vs PyTorch F.gelu(x) * y")
    print("=" * 60)

    dtype = torch.float16

    test_cases = [
        1024,
        4096,
        16384,
        65536,
        262144,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
    ]

    for n_elements in test_cases:
        print(f"\nConfiguration: N={n_elements}, dtype={dtype}")

        gate = torch.randn(n_elements, device="cuda", dtype=dtype)
        up = torch.randn(n_elements, device="cuda", dtype=dtype)

        out_triton = torch.empty_like(gate)
        out_torch = torch.empty_like(gate)

        def run_triton():
            fray.triton.gelu_mul(gate, up, out_triton)

        def run_torch():
            torch.mul(F.gelu(gate), up, out=out_torch)

        run_triton()
        run_torch()
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_torch)).item()
        is_close = torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2)

        if not is_close:
            print(
                "WARNING: correctness mismatch, "
                f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )

        t_torch_s = bench_kineto(run_torch, "torch_gelu_mul")
        t_triton_s = bench_kineto(run_triton, "triton_gelu_mul")

        bytes_per_element = gate.element_size() * 3
        total_bytes = n_elements * bytes_per_element

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(
            f"Triton GELU-Mul : {t_triton_s * 1e6:8.2f} us | "
            f"Effective Bandwidth: {triton_gbps:8.2f} GB/s"
        )
        print(
            f"PyTorch GELU-Mul: {t_torch_s * 1e6:8.2f} us | "
            f"Effective Bandwidth: {torch_gbps:8.2f} GB/s"
        )

        if t_triton_s > 0:
            print(f"Speedup         : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff        : {max_diff:.6f}")
        print(f"Mean Diff       : {mean_diff:.6f}")
        print("-" * 60)


def test_gelu_mul_2d_ffn_shape():
    print("\n" + "=" * 60)
    print("FFN-like 2D Shape Test: [M, N]")
    print("=" * 60)

    dtype = torch.float16

    test_cases = [
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (4096, 11008),
    ]

    for M, N in test_cases:
        print(f"\nConfiguration: M={M}, N={N}, dtype={dtype}")

        gate = torch.randn((M, N), device="cuda", dtype=dtype)
        up = torch.randn((M, N), device="cuda", dtype=dtype)

        out_triton = torch.empty_like(gate)
        out_torch = torch.empty_like(gate)

        def run_triton():
            fray.triton.gelu_mul(gate, up, out_triton)

        def run_torch():
            torch.mul(F.gelu(gate), up, out=out_torch)

        run_triton()
        run_torch()
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_torch)).item()
        is_close = torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2)

        if not is_close:
            print(
                "WARNING: correctness mismatch, "
                f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )

        t_torch_s = bench_kineto(run_torch, "torch_gelu_mul")
        t_triton_s = bench_kineto(run_triton, "triton_gelu_mul")

        n_elements = gate.numel()
        bytes_per_element = gate.element_size() * 3
        total_bytes = n_elements * bytes_per_element

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(
            f"Triton GELU-Mul : {t_triton_s * 1e6:8.2f} us | "
            f"Effective Bandwidth: {triton_gbps:8.2f} GB/s"
        )
        print(
            f"PyTorch GELU-Mul: {t_torch_s * 1e6:8.2f} us | "
            f"Effective Bandwidth: {torch_gbps:8.2f} GB/s"
        )

        if t_triton_s > 0:
            print(f"Speedup         : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff        : {max_diff:.6f}")
        print(f"Mean Diff       : {mean_diff:.6f}")
        print("-" * 60)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    gelu_mul_accuracy_test()

    print("\nStep 2: Running Performance Test...")
    test_gelu_mul_performance()

    print("\nStep 3: Running FFN-like 2D Shape Test...")
    test_gelu_mul_2d_ffn_shape()
