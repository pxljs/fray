import torch
import fray
from fray import bench_kineto

def test_softmax():
    print('\n' + "="*50)
    print('🚀 Performance Benchmark: Custom Softmax vs PyTorch')
    print("="*50)

    B, H, S, D = 2, 12, 4096, 128 
    x = torch.randn(B, H, S, D, dtype=torch.float, device='cuda')
    y_custom = torch.empty_like(x)
    
    # 获取总访存量 (Total Bytes Transferred)
    # Softmax 2-Pass: 2次读 (Pass1 max, Pass2 sum), 1次写 (Final write)
    # Pass 1: 读 x (1.0)
    # Pass 2: 读 x (1.0)
    # 写 y (1.0)
    # 总计 3.0 倍数据量
    total_bytes = x.numel() * x.element_size() * 3
    total_gb = total_bytes / 1e9

    def run_custom():
        # 仅测量计算函数
        fray.jit_kernels.softmax(x, y_custom)

    def run_pytorch():
        # PyTorch 的 softmax 通常是高度优化的（可能会做算子融合）
        return torch.nn.functional.softmax(x, dim=-1)

    print(f"Benchmarking Fray Softmax...")
    t_custom = bench_kineto(run_custom, 'fray_softmax')
    
    print(f"Benchmarking PyTorch Native...")
    t_pytorch = bench_kineto(run_pytorch, 'pt_softmax')

    custom_us = t_custom * 1e6
    pytorch_us = t_pytorch * 1e6
    
    custom_bw = total_gb / t_custom
    pytorch_bw = total_gb / t_pytorch

    print("\n" + "-"*30)
    print(f"Fray Softmax    : {custom_us:8.2f} us | Bandwidth: {custom_bw:7.2f} GB/s")
    print(f"PyTorch Softmax : {pytorch_us:8.2f} us | Bandwidth: {pytorch_bw:7.2f} GB/s")
    print(f"Speedup         : {pytorch_us / custom_us:8.2f}x")
    print("-"*30)

if __name__ == "__main__":
    # 验证正确性
    print("Step 1: Running Accuracy Test...")
    fray.jit_kernels.softmax_accuracy_test()
    
    # 对比性能
    print("\nStep 2: Running Performance Test...")
    test_softmax()