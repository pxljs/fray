import os
import sys
import torch

def bench_kineto(fn, kernel_names, num_tests: int = 100, flush_l2: bool = False):
    # GPU warmup
    warmup_gemm = torch.randn((4096, 4096), device='cuda')
    for _ in range(5):
        _ = warmup_gemm @ warmup_gemm
    
    # kernel warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    times = []
    
    # prepare data for cache
    if flush_l2:
        cache_flusher = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    for _ in range(num_tests):
        if flush_l2:
            cache_flusher.zero_()
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn()
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end)) # ms

    # till the pre 10% due to launch expense
    times = sorted(times)[int(num_tests * 0.1):]
    avg_time_ms = sum(times) / len(times)
    avg_time_us = avg_time_ms * 1000.0
    
    print(f"\n[Bench] {kernel_names} 真实平均耗时: {avg_time_us:.2f} us")

    is_tupled = isinstance(kernel_names, tuple)

    return tuple([avg_time_ms / 1000] * len(kernel_names)) if is_tupled else (avg_time_ms / 1000) # s

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def count_bytes(tensors):
    total = 0
    for t in tensors:
        if isinstance(t, tuple):
            total += count_bytes(t)
        else:
            total += t.numel() * t.element_size()
    return total