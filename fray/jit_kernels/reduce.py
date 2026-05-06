import torch
from .tuner import jit_tuner

includes = ('"reduce/reduce.cuh"', )
template = """
// Templated args from Python JIT call
fray::reduce_sum_c<float, {BLOCK_SIZE}>(X, y0, N);
fray::reduce_max_c<float, {BLOCK_SIZE}>(X, y1, N);
"""

def reduce_sum_max(x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor) -> None:
    N = x.shape[0]
    assert x.dtype == torch.float32 and y0.dtype == torch.float32 and y1.dtype == torch.float32

    assert N % 4 == 0, "N 必须是 4 的倍数以支持 float4 向量化加载"

    global includes, template
    
    args = (x, y0, y1, N)
    # JIT compile and runtime tunner 
    runtime = jit_tuner.compile_and_tune(
        name='reduce_sum_max',
        keys={},
        space = ({'BLOCK_SIZE': 128}, {'BLOCK_SIZE': 256}, {'BLOCK_SIZE': 512}, {'BLOCK_SIZE': 1024}),
        includes=includes,
        arg_defs=(('X', torch.float), ('y0', torch.float), ('y1', torch.float), ('N', int)),
        template=template,
        args=args
    )
    
    y0.zero_()
    y1.fill_(float('-inf'))

    runtime(*args)


def accuracy_test():
    for _ in range(1):
        torch.manual_seed(42)
        N = 4096*1024
        x = torch.randn(N, dtype=torch.float, device='cuda')
        y0 = torch.zeros(1, dtype=torch.float, device='cuda')
        y1 = torch.full((1,), float('-inf'), dtype=torch.float, device='cuda')

        # 调用我们 JIT 编译的高性能算子
        reduce_sum_max(x, y0, y1)
        
        # 获取 PyTorch 原生对照组的结果
        ref_sum = torch.sum(x)
        ref_max = torch.max(x)
        
        print("\n=== Sum Reduction ===")
        print(f"Fray Output: {y0.item()}")
        print(f"Torch Ref  : {ref_sum.item()}")
        
        print("\n=== Max Reduction ===")
        print(f"Fray Output: {y1.item()}")
        print(f"Torch Ref  : {ref_max.item()}")
        
        # 数值验证 (Sum 会有较大的累加浮点误差，所以 rtol/atol 设宽一点)
        assert torch.allclose(y0, ref_sum, rtol=1e-3, atol=1e-3), "Sum Mismatch!"
        # Max 的验证应该是绝对精确的
        assert torch.allclose(y1, ref_max), "Max Mismatch!"
        
        print("\n✅ Test passed!")
        
        # assert torch.allclose(y, y_ref, rtol=0.5, atol=0.1)

if __name__ == "__main__":
    accuracy_test()