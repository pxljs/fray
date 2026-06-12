import torch
import fray
from fray import bench_kineto

def accuracy_test():
    torch.manual_seed(42)
    B, Seq, D = 4, 1024, 4096
    epsilon = 1e-5

    x = torch.randn((B, Seq, D), dtype=torch.half, device='cuda')
    res = torch.randn((B, Seq, D), dtype=torch.half, device='cuda')
    weight = torch.randn(D, dtype=torch.half, device='cuda')
    
    res_clone = res.clone()

    # 运行自定义算子 (FP16)
    # 此时 res 会被原地更新为 x + res
    out_fray = fray.jit_kernels.fused_rmsnorm(x, res, weight, epsilon)

    # 全链路使用 FP32 计算
    # 消除 PyTorch 在 FP16 模式下不确定的累加顺序导致的误差
    x_f32 = x.to(torch.float32)
    res_f32 = res_clone.to(torch.float32)
    w_f32 = weight.to(torch.float32)

    # Residual Add -> Variance -> RMSNorm
    res_new_f32 = x_f32 + res_f32
    variance_f32 = torch.mean(res_new_f32.pow(2), dim=-1, keepdim=True)
    out_ref_f32 = res_new_f32 * torch.rsqrt(variance_f32 + epsilon) * w_f32

    # 转回 FP16 进行对比
    out_ref = out_ref_f32.to(torch.half)
    res_ref = res_new_f32.to(torch.half)

    # 精度对比
    # 对于 FP16 融合算子，atol=1e-2, rtol=1e-3 是工业界公认的合格线
    max_diff_out = torch.max(torch.abs(out_fray - out_ref)).item()
    max_diff_res = torch.max(torch.abs(res - res_ref)).item()
    
    is_out_ok = torch.allclose(out_fray, out_ref, rtol=1e-3, atol=1e-2)
    is_res_ok = torch.allclose(res, res_ref, rtol=1e-3, atol=1e-3)

    print("\n" + "="*50)
    print(f"Accuracy Check (D={D}):")
    print(f"Output Max Diff  : {max_diff_out:.6f} {'✅' if is_out_ok else '❌'}")
    print(f"Residual Max Diff: {max_diff_res:.6f} {'✅' if is_res_ok else '❌'}")
    
    if not is_out_ok:
        print(f"\nDebug Info:")
        print(f"Fray (first 4): {out_fray[0,0,:4].tolist()}")
        print(f"Ref  (first 4): {out_ref[0,0,:4].tolist()}")
    print("="*50 + "\n")


def performance_test():
    print("\n" + "="*60)
    print(" Performance Benchmark: Fray Fused RMSNorm vs PyTorch")
    print("="*60)

    epsilon = 1e-5
    
    # 模拟常见大模型的推理/训练维度 (LLaMA-3-8B 的 hidden_dim 是 4096)
    for num_tokens in (1, 2048, 8192):
        for hidden_dim in (4096, 8192):
            print(f"\nConfiguration: Tokens = {num_tokens:<4}, Hidden_Dim = {hidden_dim:<4}")
            
            x = torch.randn((num_tokens, hidden_dim), dtype=torch.half, device='cuda')
            res = torch.randn((num_tokens, hidden_dim), dtype=torch.half, device='cuda')
            weight = torch.randn(hidden_dim, dtype=torch.half, device='cuda')
            res_clone = res.clone()

            # ----------------------------------------------------
            # 计算理论读写数据量 (Memory Movement)
            # ----------------------------------------------------
            # 读 x(2B), 读 res(2B), 读 weight(2B, broadcast), 写 res(2B), 写 output(2B)
            # 总读写字节数 ≈ Tokens * Hidden_Dim * (2+2+2+2) + Hidden_Dim * 2
            bytes_moved = num_tokens * hidden_dim * 8 + hidden_dim * 2
            gb_multiplier = bytes_moved / (1024 ** 3) # 转换为 GB

            def run_fray():
                fray.jit_kernels.fused_rmsnorm(x, res, weight, epsilon)

            def run_pytorch():
                # PyTorch 原生只能通过非融合操作进行
                res_new = x + res_clone
                var = res_new.to(torch.float32).pow(2).mean(-1, keepdim=True)
                return (res_new * torch.rsqrt(var + epsilon) * weight).to(torch.half)

            # 执行测速
            t_fray = bench_kineto(run_fray, 'fused_rmsnorm_kernel')
            t_pt = bench_kineto(run_pytorch, 'add') # PyTorch 会产生一堆碎片 kernel，追踪大概率不准，通常以整体时间为准

            us_fray = t_fray * 1e6
            us_pt = t_pt * 1e6
            
            # 计算有效显存带宽 (Effective Memory Bandwidth, GB/s)
            bw_fray = gb_multiplier / t_fray if t_fray > 0 else 0
            bw_pt = gb_multiplier / t_pt if t_pt > 0 else 0

            print("-" * 60)
            print(f"Fray Fused RMSNorm : {us_fray:8.2f} us | Bandwidth: {bw_fray:6.2f} GB/s")
            print(f"PyTorch Native     : {us_pt:8.2f} us | Bandwidth: {bw_pt:6.2f} GB/s")
            if us_fray > 0:
                print(f"Speedup            : {us_pt / us_fray:8.2f}x")
            print("-" * 60)

if __name__ == "__main__":
    accuracy_test()
    performance_test()