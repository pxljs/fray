import torch
import fray
from fray import bench_kineto

def precompute_rope_params(max_seq_len, dim, theta=10000.0):
    """生成符合 Llama 规范的 cos/sin 表"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq) # [max_seq_len, dim // 2]
    return torch.cos(freqs).cuda(), torch.sin(freqs).cuda()

def torch_rope_reference(q, k, cos, sin, offsets):
    """PyTorch 实现的 RoPE (前半部分与后半部分旋转)"""
    def apply_rotary(x, cos, sin, offsets):
        # x: [num_tokens, num_heads, head_dim]
        d = x.shape[-1]
        half_d = d // 2
        # 获取每个 token 对应的 cos/sin
        c = cos[offsets].unsqueeze(1) # [num_tokens, 1, half_d]
        s = sin[offsets].unsqueeze(1) # [num_tokens, 1, half_d]
        
        x_first = x[..., :half_d]
        x_second = x[..., half_d:]
        
        # 旋转公式
        out_first = x_first * c - x_second * s
        out_second = x_first * s + x_second * c
        return torch.cat([out_first, out_second], dim=-1)

    return apply_rotary(q, cos, sin, offsets), apply_rotary(k, cos, sin, offsets)

def accuracy_test():
    torch.manual_seed(42)
    num_tokens = 1024
    num_heads = 32
    num_kv_heads = 8 # GQA
    head_dim = 128
    
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.half, device='cuda')
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device='cuda')
    offsets = torch.randint(0, 512, (num_tokens,), dtype=torch.int32, device='cuda')
    
    cos, sin = precompute_rope_params(2048, head_dim)
    
    # 拷贝一份用于对比
    q_ref, k_ref = q.clone(), k.clone()
    
    # 运行 Fray 算子
    fray.jit_kernels.fused_rope(q, k, cos, sin, offsets)
    
    # 运行参考实现
    q_expected, k_ref_expected = torch_rope_reference(q_ref, k_ref, cos, sin, offsets)

    q_expected = q_expected.to(torch.half)
    k_ref_expected = k_ref_expected.to(torch.half)
    
    # 校验
    max_diff_q = torch.max(torch.abs(q - q_expected)).item()
    max_diff_k = torch.max(torch.abs(k - k_ref_expected)).item()
    
    print("\n" + "="*50)
    print(f"Accuracy Check (HEAD_DIM={head_dim}):")
    print(f"Q Max Diff: {max_diff_q:.6f} {'✅' if max_diff_q < 5e-3 else '❌'}")
    print(f"K Max Diff: {max_diff_k:.6f} {'✅' if max_diff_k < 5e-3 else '❌'}")
    print("="*50 + "\n")

def performance_test():
    print("\n" + "="*60)
    print(" Performance Benchmark: Fray Fused RoPE (Memory Bandwidth)")
    print("="*60)
    
    configs = [
        # 经典模型 (如 GPT-3 风格 MHA)
        (1024, 32, 32, 64),
        (2048, 32, 32, 128),
        
        # 现代大模型 (如 LLaMA-2/3 8B 风格 GQA)
        (1024, 32, 8, 128),
        (4096, 32, 8, 128),
        (8192, 32, 8, 128), # 长文本
        
        # 超大模型 (如 LLaMA-3 70B 风格 GQA)
        (2048, 64, 8, 128),
    ]

    for num_tokens, num_heads, num_kv_heads, head_dim in configs:
        cos, sin = precompute_rope_params(8192, head_dim)
        q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.half, device='cuda')
        k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device='cuda')
        offsets = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
        
        # 计算理论访存量 (GB)
        # Q: 读(2B) + 写(2B); K: 读(2B) + 写(2B); Cos/Sin: 读(4B*2)
        # 注意 Cos/Sin 在 Head 间共享，通常在 L2 Cache 命中
        bytes_moved = num_tokens * (num_heads + num_kv_heads) * head_dim * 4 + num_tokens * head_dim * 4
        gb_multiplier = bytes_moved / (1024 ** 3)

        def run_fray():
            fray.jit_kernels.fused_rope(q, k, cos, sin, offsets)

        t_fray = bench_kineto(run_fray, 'fused_rope_kernel')
        bw = gb_multiplier / t_fray if t_fray > 0 else 0

        print(f"num_tokens:{num_tokens:<6}num_heads:{num_heads:<4}num_kv_heads:{num_kv_heads:<4}head_dim:{head_dim:<4}  "
              f"time:{t_fray*1e6:8.2f}us  bw:{bw:8.2f}GB/s")

if __name__ == "__main__":
    accuracy_test()
    performance_test()