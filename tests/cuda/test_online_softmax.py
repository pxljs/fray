import torch
import fray
from fray import bench_kineto

def test_online_softmax():
    print('\n' + "="*50)
    print(' Performance Benchmark: Online Softmax vs PyTorch')
    print("="*50)

    # Prepare test data 
    M, D = 16384, 512 
    x = torch.randn(M, D, dtype=torch.float, device='cuda')
    y_custom = torch.empty_like(x)
    
    # Total Bytes Transferred
    # Online Softmax : one read , one write
    total_bytes = x.numel() * x.element_size() * 2
    total_gb = total_bytes / 1e9

    # test fray online_softmax
    def run_online():
        fray.jit_kernels.online_softmax(x, y_custom)

    # test pytorch softmax
    def run_pytorch():
        return torch.nn.functional.softmax(x, dim=-1)

    print(f"Benchmarking Fray Online Softmax...")
    t_online = bench_kineto(run_online, 'fray_online_softmax')
    
    print(f"Benchmarking PyTorch Native...")
    t_pytorch = bench_kineto(run_pytorch, 'pt_softmax')

    online_us = t_online * 1e6
    pytorch_us = t_pytorch * 1e6
    
    # Cal Effective Bandwidth
    online_bw = total_gb / t_online
    pytorch_bw = total_gb / t_pytorch

    print("\n" + "-"*30)
    print(f"Fray Online Softmax: {online_us:8.2f} us | Bandwidth: {online_bw:7.2f} GB/s")
    print(f"PyTorch Softmax    : {pytorch_us:8.2f} us | Bandwidth: {pytorch_bw:7.2f} GB/s")
    print(f"Speedup            : {pytorch_us / online_us:8.2f}x")
    print("-"*30)

if __name__ == "__main__":
    # verify correctness
    print("Step 1: Running Accuracy Test...")
    fray.jit_kernels.online_softmax_accuracy_test()
    
    # compare performance
    print("\nStep 2: Running Performance Test...")
    test_online_softmax()