import torch
import fray
from fray import bench_kineto

def test_fp16_gemm():
    print('\n' + "="*50)
    print(' Performance Benchmark: Fray FP16 GEMM vs PyTorch (cuBLAS)')
    print("="*50)

    for M in (256, 4096):
         for K, N in [(5120, 5120), (1536, 24576), (512, 32768), (16384, 7168), (7168, 2048), (2048, 7168)]:  
            # A 是行主序 (Row-Major)
            a = torch.randn((M, K), dtype=torch.half, device='cuda').contiguous()
            b = torch.randn((N, K), dtype=torch.half, device='cuda')
            
            c_custom = torch.zeros((M, N), dtype=torch.half, device='cuda').contiguous()
            
            # 2. 计算 TFLOPS 指标基准
            # GEMM FLOPs = 2 * M * N * K
            total_flops = 2.0 * M * N * K
            # 转换为 TFLOPS 的乘子: (FLOPs / 1e12)
            tflops_multiplier = total_flops / 1e12

            # 3. 定义测试闭包
            def run_custom():
                fray.jit_kernels.fp16_gemm(a, b, c_custom)

            def run_pytorch():
                return torch.matmul(a, b.t())

            # 4. 执行 Kineto 测速
            print(f"Benchmarking Fray CuTe GEMM...")
            t_custom = bench_kineto(run_custom, 'fray_cute_gemm')
            
            print(f"Benchmarking PyTorch Native (cuBLAS)...")
            t_pytorch = bench_kineto(run_pytorch, 'pt_cublas_gemm')

            # 5. 指标计算
            custom_us = t_custom * 1e6
            pytorch_us = t_pytorch * 1e6
            
            # TFLOPS = (FLOPs / 1e12) / 秒
            custom_tflops = tflops_multiplier / t_custom
            pytorch_tflops = tflops_multiplier / t_pytorch

            print("\n" + "-"*45)
            print(f"Fray CuTe GEMM : {custom_us:8.2f} us | Compute: {custom_tflops:7.2f} TFLOPS")
            print(f"PyTorch cuBLAS : {pytorch_us:8.2f} us | Compute: {pytorch_tflops:7.2f} TFLOPS")
            print(f"Speedup        : {pytorch_us / custom_us:8.2f}x")
            print("-" * 45)

if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    fray.jit_kernels.fp16_gemm_accuracy_test()
    
    # 对比性能
    print("\nStep 2: Running Performance Test...")
    test_fp16_gemm()