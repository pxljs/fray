import torch
import torch.nn.functional as F

import fray
from fray import bench_kineto

def flash_attn_accuracy_test():
    torch.manual_seed(42)
    # 使用较小规模进行精度验证，节省时间
    B, H, N, D = 2, 4, 1024, 128
    
    # 统一使用 [B, H, N, D] 布局
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.half)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.half)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.half)
    out_fray = torch.zeros_like(q)
    
    # 1. 运行自定义算子
    fray.jit_kernels.flash_attn_cute(q, k, v, out_fray)
    
    # 2. 运行 PyTorch 原生最优算子 (底层自动调用 FlashAttention)
    # 注意：PyTorch 默认需要 query, key, value，并且天然支持 [B, H, N, D]
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # 3. 精度对比
    max_diff = torch.max(torch.abs(out_fray - out_ref)).item()
    is_close = max_diff < 1e-2
    
    print("\n" + "="*50)
    print(f"Fray Output Sample: {out_fray[0, 0, 0, :4].tolist()}")
    print(f"Torch Ref Sample  : {out_ref[0, 0, 0, :4].tolist()}")
    print(f"Accuracy Check: {'✅ PASS' if is_close else '❌ FAIL'} (Max Diff: {max_diff:.6f})")
    print("="*50 + "\n")


def test_flash_attn():
    print('\n' + "="*50)
    print(' Performance Benchmark: Fray Flash Attention vs PyTorch (SDPA)')
    print("="*50)

    head_dim = 128
    batch_size = 16  # 显存不够可以调小为 8

    for seq_len in (1024, 2048, 4096):
        for num_heads in (12, 16, 24):
            print(f"\nConfiguration: B={batch_size}, N={seq_len}, H={num_heads}, D={head_dim}")
            
            # 1. 准备数据 (全部保持 [B, H, N, D])
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.half)
            out_fray = torch.zeros_like(q)

            # 2. 计算 TFLOPS 指标基准
            # FLOPs = 4 * B * H * N^2 * D
            total_flops = 4.0 * batch_size * num_heads * (seq_len ** 2) * head_dim
            tflops_multiplier = total_flops / 1e12

            # 3. 定义测试闭包
            def run_custom():
                fray.jit_kernels.flash_attn_cute(q, k, v, out_fray)

            def run_pytorch():
                # 强制 PyTorch 尝试使用底层的 FlashAttention 后端
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=False)

            # 4. 执行 Kineto 测速
            print(f"Benchmarking Fray CuTe Flash Attention...")
            t_custom = bench_kineto(run_custom, 'flash_attn_cute_kernel')
            
            print(f"Benchmarking PyTorch Native (SDPA)...")
            # PyTorch 原生 FlashAttention 内核通常包含 'flash' 字样
            t_pytorch = bench_kineto(run_pytorch, 'flash')

            # 5. 指标计算
            custom_us = t_custom * 1e6
            pytorch_us = t_pytorch * 1e6
            
            custom_tflops = tflops_multiplier / t_custom if t_custom > 0 else 0
            pytorch_tflops = tflops_multiplier / t_pytorch if t_pytorch > 0 else 0

            print("-" * 50)
            print(f"Fray CuTe FA   : {custom_us:8.2f} us | Compute: {custom_tflops:7.2f} TFLOPS")
            print(f"PyTorch SDPA   : {pytorch_us:8.2f} us | Compute: {pytorch_tflops:7.2f} TFLOPS")
            if custom_us > 0:
                print(f"Speedup        : {pytorch_us / custom_us:8.2f}x")
            print("-" * 50)


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    flash_attn_accuracy_test()
    
    print("\nStep 2: Running Performance Test...")
    test_flash_attn()