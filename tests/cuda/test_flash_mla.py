import torch
import torch.nn.functional as F
import math

import fray
from fray import bench_kineto

def accuracy_test():
    torch.manual_seed(42)
    
    # 模拟 MLA 解码阶段结构 (H_q = 128, Nope = 512, Rope = 64)
    B, H_q, Seq, Block_Size = 2, 128, 2048, 64
    d_nope, d_rope = 512, 64
    D_q = d_nope + d_rope # 576

    seq_lens = torch.tensor([2051, 1037], device='cuda', dtype=torch.int32)
    
    # Q: [B, 1, H_q, D_q]
    q = torch.randn(B, 1, H_q, D_q, device='cuda', dtype=torch.half)
    
    # 分配物理分页池 KV_cache
    # 每个物理页包含 64 个 Tokens
    max_blocks_0 = math.ceil(seq_lens[0].item() / Block_Size)
    max_blocks_1 = math.ceil(seq_lens[1].item() / Block_Size)
    max_blocks_per_seq = max(max_blocks_0, max_blocks_1)
    
    total_pages = max_blocks_0 + max_blocks_1
    KV_cache = torch.randn(total_pages, Block_Size, D_q, device='cuda', dtype=torch.half)
    
    # 物理页表映射：为 Batch 0 分配前 32 个物理页，为 Batch 1 分配后 16 个物理页
    block_table = torch.zeros(B, max_blocks_per_seq, device='cuda', dtype=torch.int32)
    block_table[0, :max_blocks_0] = torch.arange(0, max_blocks_0, dtype=torch.int32)
    block_table[1, :max_blocks_1] = torch.arange(max_blocks_0, total_pages, dtype=torch.int32)
    
    out_fray = torch.zeros(B, H_q, d_nope, device='cuda', dtype=torch.half)
    scale = 1.0 / math.sqrt(d_nope + d_rope) # 1 / sqrt(576)
    
    # 1. 运行 Fray 自定义 Paged FlashMLA 解码
    fray.jit_kernels.flash_mla(q, KV_cache, block_table, seq_lens, out_fray, scale)
    
    # 2. 运行 PyTorch 官方参考值
    # 强制转为 FP32 计算以消除 FP16 累加带来的正常精度漂移
    out_ref = torch.zeros(B, H_q, d_nope, device='cuda', dtype=torch.half)
    
    for b in range(B):
        length = seq_lens[b].item()
        num_blocks = math.ceil(length / Block_Size)
        
        # 重构当前序列的物理连续表征
        kv_pages = []
        for t in range(num_blocks):
            page_idx = block_table[b, t].item()
            kv_pages.append(KV_cache[page_idx]) # [64, 576]
            
        full_kv = torch.cat(kv_pages, dim=0)[:length].to(torch.float32) # [Seq, 576]
        c_kv = full_kv[:, :d_nope] # [Seq, 512]
        k_rope = full_kv[:, d_nope:] # [Seq, 64]
        
        q_nope = q[b, 0, :, :d_nope].to(torch.float32) # [H_q, 512]
        q_rope = q[b, 0, :, d_nope:].to(torch.float32) # [H_q, 64]
        
        # 计算注意力得分 S = S_nope + S_rope
        S_nope = torch.matmul(q_nope, c_kv.t()) # [H_q, Seq]
        S_rope = torch.matmul(q_rope, k_rope.t()) # [H_q, Seq]
        S = (S_nope + S_rope) * scale
        
        # Softmax
        P = F.softmax(S, dim=-1) # [H_q, Seq]
        
        # 加权求和
        O_b = torch.matmul(P, c_kv) # [H_q, 512]
        out_ref[b] = O_b.to(torch.half)
        
    # 校验对比
    max_diff = torch.max(torch.abs(out_fray - out_ref)).item()
    is_close = max_diff < 1e-2
    
    print("\n" + "="*60)
    print(f"Accuracy Check (FlashMLA Paged Decoding):")
    print(f"Fray Output Sample: {out_fray[0, 0, :4].tolist()}")
    print(f"Torch Ref Sample  : {out_ref[0, 0, :4].tolist()}")
    print(f"Result: {'✅ PASS' if is_close else '❌ FAIL'} (Max Diff: {max_diff:.6f})")
    print("="*60 + "\n")


# 性能基准测试
def performance_test():
    print("\n" + "="*80)
    print(" Performance Benchmark: Fray FlashMLA vs PyTorch Native")
    print("="*80)
    
    d_nope, d_rope, Block_Size = 512, 64, 64
    D_q = d_nope + d_rope # 576
    H_q = 128
    
    # 模拟真实 DeepSeek-V3 / R1 推理场景配置 (Batch, Seq_Len)
    configs = [
        (1, 4096),   # 单用户，短上下文
        (1, 16384),  # 单用户，长上下文
        (8, 8192),   # 中并发 Batch，长上下文
        (16, 4096),  # 高并发 Batch
    ]

    print(f"{'B':<4} | {'H_q':<4} | {'SeqLen':<7} || "
          f"{'Fray (us)':<12} | {'Fray (GB/s)':<12} || "
          f"{'PyTorch (us)':<12} | {'Speedup':<8}")
    print("-" * 80)

    for B, Seq in configs:
        scale = 1.0 / math.sqrt(d_nope + d_rope)
        
        q = torch.randn(B, 1, H_q, D_q, device='cuda', dtype=torch.half)
        seq_lens = torch.full((B,), Seq, device='cuda', dtype=torch.int32)
        
        # 分配物理分页池
        max_blocks_per_seq = math.ceil(Seq / Block_Size)
        total_pages = B * max_blocks_per_seq
        KV_cache = torch.randn(total_pages, Block_Size, D_q, device='cuda', dtype=torch.half)
        
        # 页表映射
        block_table = torch.arange(0, total_pages, dtype=torch.int32, device='cuda').view(B, max_blocks_per_seq)
        
        out_fray = torch.zeros(B, H_q, d_nope, device='cuda', dtype=torch.half)

        # 计算理论显存移动量 (GB)
        # 读 Q: B * H_q * 1 * 576 * 2 bytes
        # 读 KV: B * Seq * 576 * 2 bytes (潜变量 c_KV + 位置向量 k_RoPE)
        # 写 O: B * H_q * 512 * 2 bytes
        bytes_moved = (B * H_q * D_q * 2) + (B * Seq * D_q * 2) + (B * H_q * d_nope * 2)
        gb_multiplier = bytes_moved / (1024 ** 3)

        def run_fray():
            fray.jit_kernels.flash_mla(q, KV_cache, block_table, seq_lens, out_fray, scale)

        # 向量化高度优化的 PyTorch 参考实现（包含页表物理连续合并开销，以确保对比公平）
        def run_pytorch():
            # 1. 页表物理合并
            pages = block_table.view(-1)
            full_pages = KV_cache[pages] # [B * Max_Blocks, 64, 576]
            full_kv = full_pages.view(B, max_blocks_per_seq * Block_Size, D_q)
            
            # 2. 提取子矩阵
            c_kv = full_kv[:, :Seq, :d_nope] # [B, Seq, 512]
            k_rope = full_kv[:, :Seq, d_nope:] # [B, Seq, 64]
            
            q_nope_flat = q[:, 0, :, :d_nope].unsqueeze(2) # [B, H_q, 1, 512]
            q_rope_flat = q[:, 0, :, d_nope:].unsqueeze(2) # [B, H_q, 1, 64]
            
            # 3. 注意力矩阵点积
            S_nope = torch.matmul(q_nope_flat, c_kv.unsqueeze(1).transpose(-1, -2)) # [B, H_q, 1, Seq]
            S_rope = torch.matmul(q_rope_flat, k_rope.unsqueeze(1).transpose(-1, -2)) # [B, H_q, 1, Seq]
            
            S = (S_nope + S_rope) * scale
            P = F.softmax(S, dim=-1) # [B, H_q, 1, Seq]
            
            # 4. 加权求和输出
            O_b = torch.matmul(P, c_kv.unsqueeze(1)) # [B, H_q, 1, 512]
            O = O_b.transpose(1, 2).squeeze(1) # [B, H_q, 512]
            return O

        t_fray = bench_kineto(run_fray, 'flash_mla_paged_decoding') 
        t_pt = bench_kineto(run_pytorch, 'torch_mla_ref') 

        # 计算物理带宽
        bw_fray = gb_multiplier / t_fray if t_fray > 0 else 0
        
        us_fray = t_fray * 1e6
        us_pt = t_pt * 1e6
        speedup = us_pt / us_fray if us_fray > 0 else 0

        print(f"{B:<4} | {H_q:<4} | {Seq:<7} || "
              f"{us_fray:8.1f} us  | {bw_fray:8.1f} GB/s || "
              f"{us_pt:8.1f} us  | {speedup:6.2f}x")

if __name__ == "__main__":
    accuracy_test()
    performance_test()
