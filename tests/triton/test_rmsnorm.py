import torch

import fray
from fray import bench_kineto


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x_f32 = x.float()
    weight_f32 = weight.float()

    variance = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)

    out = x_f32 * rstd * weight_f32
    return out.to(x.dtype)


def rmsnorm_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton RMSNorm vs PyTorch RMSNorm")
    print("=" * 60)

    eps = 1e-6
    dtype = torch.float16

    test_cases = [
        (1, 1024),
        (1024, 1024),
        (1024, 1024),
        (1024, 2048),
        (1024, 4096),
        (1024, 8192),
        (1024, 11008),
        (1024, 4096),
    ]

    for M, N in test_cases:
        x = torch.randn((M, N), device="cuda", dtype=dtype)
        weight = torch.randn((N,), device="cuda", dtype=dtype)
        out_triton = torch.empty_like(x)

        fray.triton.rmsnorm(x, weight, out_triton, eps)

        out_ref = torch_rmsnorm(x, weight, eps)

        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()
        is_close = torch.allclose(out_triton, out_ref, rtol=1e-2, atol=1e-2)

        print(f"\nShape = [{M}, {N}], dtype={dtype}")
        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print(f"Accuracy Check      : {'✅ PASS' if is_close else '❌ FAIL'}")
        print(f"Max Diff            : {max_diff:.6f}")
        print(f"Mean Diff           : {mean_diff:.6f}")

    print("=" * 60 + "\n")


def test_rmsnorm_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton RMSNorm vs PyTorch RMSNorm")
    print("=" * 60)

    eps = 1e-6
    dtype = torch.float16

    test_cases = [
        (1, 4096),
        (128, 1024),
        (512, 1024),
        (1024, 1024),
        (1024, 2048),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (4096, 8192),
        (4096, 11008),
        (8192, 4096),
    ]

    for M, N in test_cases:
        print(f"\nConfiguration: M={M}, N={N}, dtype={dtype}, eps={eps}")

        x = torch.randn((M, N), device="cuda", dtype=dtype)
        weight = torch.randn((N,), device="cuda", dtype=dtype)

        out_triton = torch.empty_like(x)
        out_torch = torch.empty_like(x)

        def run_triton():
            fray.triton.rmsnorm(x, weight, out_triton, eps)

        def run_torch():
            out_torch.copy_(torch_rmsnorm(x, weight, eps))

        # 显式触发初始化 / Triton JIT，避免污染正式计时
        run_triton()
        run_torch()
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_torch)).item()
        is_close = torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2)

        if not is_close:
            print(f"WARNING: correctness mismatch, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        t_torch_s = bench_kineto(run_torch, "torch_rmsnorm")
        t_triton_s = bench_kineto(run_triton, "triton_rmsnorm")

        n_elements = x.numel()

        # Triton RMSNorm 逻辑有效访存量：
        # read x + read weight + write out
        #
        # PyTorch baseline 实际访存会更高，因为它有多个中间 tensor。
        # 这里统计 effective bandwidth，方便和 Triton 逻辑算子对比。
        bytes_per_element = x.element_size() * 3
        total_bytes = n_elements * bytes_per_element

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(f"Triton RMSNorm : {t_triton_s * 1e6:8.2f} us | Effective Bandwidth: {triton_gbps:8.2f} GB/s")
        print(f"PyTorch RMSNorm: {t_torch_s * 1e6:8.2f} us | Effective Bandwidth: {torch_gbps:8.2f} GB/s")

        if t_triton_s > 0:
            print(f"Speedup        : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff       : {max_diff:.6f}")
        print(f"Mean Diff      : {mean_diff:.6f}")
        print("-" * 60)


def test_rmsnorm_3d_shape():
    print("\n" + "=" * 60)
    print("3D Shape Test: [B, S, H]")
    print("=" * 60)

    eps = 1e-6
    dtype = torch.float16

    test_cases = [
        (1, 128, 1024),
        (2, 512, 2048),
        (4, 1024, 4096),
        (8, 1024, 4096),
        (4, 2048, 8192),
    ]

    for B, S, H in test_cases:
        print(f"\nConfiguration: B={B}, S={S}, H={H}, dtype={dtype}, eps={eps}")

        x = torch.randn((B, S, H), device="cuda", dtype=dtype)
        weight = torch.randn((H,), device="cuda", dtype=dtype)

        out_triton = torch.empty_like(x)
        out_torch = torch.empty_like(x)

        def run_triton():
            fray.triton.rmsnorm(x, weight, out_triton, eps)

        def run_torch():
            out_torch.copy_(torch_rmsnorm(x, weight, eps))

        run_triton()
        run_torch()
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
        mean_diff = torch.mean(torch.abs(out_triton - out_torch)).item()
        is_close = torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2)

        if not is_close:
            print(f"WARNING: correctness mismatch, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        t_torch_s = bench_kineto(run_torch, "torch_rmsnorm")
        t_triton_s = bench_kineto(run_triton, "triton_rmsnorm")

        n_elements = x.numel()
        bytes_per_element = x.element_size() * 3
        total_bytes = n_elements * bytes_per_element

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(f"Triton RMSNorm : {t_triton_s * 1e6:8.2f} us | Effective Bandwidth: {triton_gbps:8.2f} GB/s")
        print(f"PyTorch RMSNorm: {t_torch_s * 1e6:8.2f} us | Effective Bandwidth: {torch_gbps:8.2f} GB/s")

        if t_triton_s > 0:
            print(f"Speedup        : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff       : {max_diff:.6f}")
        print(f"Mean Diff      : {mean_diff:.6f}")
        print("-" * 60)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    rmsnorm_accuracy_test()

    print("\nStep 2: Running Performance Test...")
    test_rmsnorm_performance()

    print("\nStep 3: Running 3D Shape Test...")
    test_rmsnorm_3d_shape()