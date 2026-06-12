import torch
import torch.nn.functional as F

import fray
from fray import bench_kineto
from torch.nn.attention import SDPBackend, sdpa_kernel

def accuracy_test():
    torch.manual_seed(42)
    
    # 小规模 GQA 结构 (Group = H_q / H_kv = 4)
    B, H_q, H_kv, Seq, D = 2, 16, 4, 2048, 128
    G = H_q // H_kv
    
    # Q: [B, H_q, 1, D]
    q = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.half)
    # K, V: [B, H_kv, Seq, D]
    k = torch.randn(B, H_kv, Seq, D, device='cuda', dtype=torch.half)
    v = torch.randn(B, H_kv, Seq, D, device='cuda', dtype=torch.half)
    
    out_fray = torch.zeros_like(q)
    
    # 运行 Fray 自定义 Flash Decoding
    fray.jit_kernels.flash_decoding(q, k, v, out_fray)
    
    # 运行 PyTorch 官方参考值
    # 强制转为 FP32 计算以消除 FP16 树状规约带来的正常精度漂移
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    
    # GQA 展开。将 1 个 KV 头物理复制成 4 个，以对齐 Q 的头数
    k_expanded = k_f32.repeat_interleave(G, dim=1).contiguous()
    v_expanded = v_f32.repeat_interleave(G, dim=1).contiguous()
    
    # 运行 PyTorch 原生 SDPA
    out_ref_f32 = F.scaled_dot_product_attention(q_f32, k_expanded, v_expanded, is_causal=False)
    out_ref = out_ref_f32.to(torch.half)
    
    # 校验对比
    max_diff = torch.max(torch.abs(out_fray - out_ref)).item()
    # Decode 阶段累加极长，FP16 下 1e-2 是正常的绝对误差阈值
    is_close = max_diff < 1e-2
    
    print("\n" + "="*60)
    print(f"Accuracy Check (Flash Decoding Split-K):")
    print(f"Fray Output Sample: {out_fray[0, 0, 0, :4].tolist()}")
    print(f"Torch Ref Sample  : {out_ref[0, 0, 0, :4].tolist()}")
    print(f"Result: {'✅ PASS' if is_close else '❌ FAIL'} (Max Diff: {max_diff:.6f})")
    print("="*60 + "\n")


# 性能基准测试
def performance_test():
    print("\n" + "="*80)
    print(" Performance Benchmark: Fray Flash Decoding vs PyTorch Native")
    print("="*80)
    
    D = 128
    
   # 模拟真实 LLM Decode 场景配置 (Batch, H_q, H_kv, Seq_Len)
    # 以 LLaMA-3 8B 为例: H_q=32, H_kv=8
    configs = [
        (1, 32, 8, 4096),   # 单用户，短上下文
        (1, 32, 8, 32768),  # 单用户，极长上下文
        (16, 32, 8, 8192),  # 高并发 Batch，长上下文
        (32, 32, 8, 4096),  # 极限并发 Batch
    ]

    print(f"{'B':<4} | {'H_q':<4} | {'H_kv':<4} | {'SeqLen':<7} || "
          f"{'Fray (us)':<12} | {'Fray (GB/s)':<12} || "
          f"{'PyTorch (us)':<12} | {'Speedup':<8}")
    print("-" * 80)

    for B, H_q, H_kv, Seq in configs:
        G = H_q // H_kv
        
        q = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.half)
        k = torch.randn(B, H_kv, Seq, D, device='cuda', dtype=torch.half)
        v = torch.randn(B, H_kv, Seq, D, device='cuda', dtype=torch.half)
        out_fray = torch.zeros_like(q)

        k_expanded = k.repeat_interleave(G, dim=1).contiguous()
        v_expanded = v.repeat_interleave(G, dim=1).contiguous()

        # 计算理论显存读写量 (GB)
        # 读 Q: B * H_q * 1 * D * 2 bytes
        # 读 K, V: B * H_kv * Seq * D * 2 bytes * 2
        # 写 O: B * H_q * 1 * D * 2 bytes
        bytes_moved = (B * H_q * D * 4) + (B * H_kv * Seq * D * 4)
        gb_multiplier = bytes_moved / (1024 ** 3)

        def run_fray():
            fray.jit_kernels.flash_decoding(q, k, v, out_fray)

        def run_pytorch():
            # 允许尝试 Flash Attention 或 Math 后端
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=False)


        t_fray = bench_kineto(run_fray, 'flash_decoding') 
        t_pt = bench_kineto(run_pytorch, 'flash_fwd') 

        # 计算带宽
        bw_fray = gb_multiplier / t_fray if t_fray > 0 else 0
        
        us_fray = t_fray * 1e6
        us_pt = t_pt * 1e6
        speedup = us_pt / us_fray if us_fray > 0 else 0

        print(f"{B:<4} | {H_q:<4} | {H_kv:<4} | {Seq:<7} || "
              f"{us_fray:8.1f} us  | {bw_fray:8.1f} GB/s || "
              f"{us_pt:8.1f} us  | {speedup:6.2f}x")

if __name__ == "__main__":
    accuracy_test()
    performance_test()