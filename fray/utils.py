import os
import sys
import torch
import torch.distributed as dist

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()

def bench_kineto(fn, kernel_names, num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: str = None, barrier_comm_profiling: bool = False, flush_l2: bool = False):

    # 1. 预热
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    # 2. 测速
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_tests):
        if flush_l2:
            _ = torch.empty(int(128e6 // 4), dtype=torch.int, device='cuda').zero_()
        fn()
    end.record()
    torch.cuda.synchronize()
    
    # 3. 计算时间 (微秒 us)
    avg_time_us = (start.elapsed_time(end) * 1000.0) / num_tests
    
    print(f"\n[Bench] {kernel_names} 耗时: {avg_time_us:.2f} us")

    is_tupled = isinstance(kernel_names, tuple)
    return tuple([avg_time_us/1000] * len(kernel_names)) if is_tupled else (avg_time_us/1000)

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