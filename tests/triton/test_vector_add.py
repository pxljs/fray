import torch

import fray
from fray import bench_kineto

def vector_add_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton Vector Add vs PyTorch Native +")
    print("=" * 60)

    test_cases = [
        1024,
        4096,
        1024 * 1024,
        1024 * 1024 + 123,
    ]

    for n_elements in test_cases:
        x = torch.randn(n_elements, device="cuda", dtype=torch.float16)
        y = torch.randn(n_elements, device="cuda", dtype=torch.float16)
        out_triton = torch.empty_like(x)

        fray.triton.vector_add(x, y, out_triton)
        out_ref = x + y

        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
        is_close = torch.allclose(out_triton, out_ref, rtol=1e-3, atol=1e-3)

        print(f"\nN = {n_elements}")
        print(f"Triton Output Sample: {out_triton[:4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[:4].tolist()}")
        print(f"Accuracy Check      : {'✅ PASS' if is_close else '❌ FAIL'}")
        print(f"Max Diff            : {max_diff:.6f}")

    print("=" * 60 + "\n")


def test_vector_add_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton Vector Add vs PyTorch Native Add")
    print("=" * 60)

    dtype = torch.float16

    # 这里覆盖从小规模到大规模的 elementwise 场景
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

        x = torch.randn(n_elements, device="cuda", dtype=dtype)
        y = torch.randn(n_elements, device="cuda", dtype=dtype)

        out_triton = torch.empty_like(x)
        out_torch = torch.empty_like(x)

        def run_triton():
            fray.triton.vector_add(x, y, out_triton)

        def run_torch():
            torch.add(x, y, out=out_torch)

        run_triton()
        run_torch()
        torch.cuda.synchronize()

        t_torch_s = bench_kineto(run_torch, "torch_add")
        t_triton_s = bench_kineto(run_triton, "triton_vector_add")

        # vector add 访存量：
        # read x + read y + write out
        bytes_per_element = x.element_size() * 3
        total_bytes = n_elements * bytes_per_element

        triton_gbps = total_bytes / (t_triton_s ) / 1e9
        torch_gbps = total_bytes / (t_torch_s ) / 1e9

        print("-" * 60)
        print(f"Triton Vector Add : {t_triton_s * 1e6:8.2f} us | Bandwidth: {triton_gbps:8.2f} GB/s")
        print(f"PyTorch Native Add: {t_torch_s * 1e6:8.2f} us | Bandwidth: {torch_gbps:8.2f} GB/s")

        if t_triton_s > 0:
            print(f"Speedup           : {t_torch_s / t_triton_s:8.2f}x")
        print("-" * 60)



if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    vector_add_accuracy_test()

    print("\nStep 2: Running Performance Test...")
    test_vector_add_performance()