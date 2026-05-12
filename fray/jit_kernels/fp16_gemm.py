import torch
from .tuner import jit_tuner

includes = ('"gemm/fp16_gemm.cuh"', )
template = """
fray::fp16_gemm_c<{BLOCK_M}, {BLOCK_N}, {NUM_STAGES}>(M, N, K, A, B, C, stream);
"""

def fp16_gemm(a: torch.Tensor, b: torch.Tensor, c:torch.Tensor) -> None:
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, "K dimension mismatch"
    assert a.dtype == torch.half and b.dtype == torch.half and c.dtype == torch.half

    stream = torch.cuda.current_stream()

    global includes, template
    
    args = (M, N, K, a, b, c, stream)

    # JIT compile and runtime tunner 
    runtime = jit_tuner.compile_and_tune(
        name='fp16_gemm',
        keys={'BLOCK_M':128, 'BLOCK_N':128, 'NUM_STAGES':3},
        space = (
            {'BLOCK_M':128, 'BLOCK_N':256, 'NUM_STAGES':3},
            {'BLOCK_M':256, 'BLOCK_N':128, 'NUM_STAGES':3},
            {'BLOCK_M':128, 'BLOCK_N':128, 'NUM_STAGES':4},
            {'BLOCK_M':64, 'BLOCK_N':128, 'NUM_STAGES':4},
            {'BLOCK_M':128, 'BLOCK_N':64, 'NUM_STAGES':4},
            {'BLOCK_M':64, 'BLOCK_N':64, 'NUM_STAGES':3},
            {'BLOCK_M':32, 'BLOCK_N':32, 'NUM_STAGES':5},
        ),
        includes=includes,
        arg_defs=(('M', int), ('N', int), ('K', int),
                  ('A', torch.half), ('B', torch.half), ('C', torch.half),
                  ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )

    runtime(*args)


def accuracy_test():
    torch.manual_seed(42)

    M, N, K = 4096, 4096, 4096
    
    # 初始化数据 (FP16 输入)
    a = torch.randn((M, K), dtype=torch.half, device='cuda')
    b = torch.randn((N, K), dtype=torch.half, device='cuda')
    
    c_custom = torch.zeros((M, N), dtype=torch.half, device='cuda')
    
    fp16_gemm(a, b, c_custom)
    
    c_ref = torch.matmul(a, b.t())
    
    print("\n" + "="*50)
    print(f"Fray Output Sample: {c_custom[0, :4].tolist()}")
    print(f"Torch Ref Sample  : {c_ref[0, :4].tolist()}")
    
    # Tensor Core 计算会有不可避免的舍入误差，atol 给 1e-2 是合理的
    is_close = torch.allclose(c_custom, c_ref, rtol=1e-2, atol=1e-2)
    print(f"Accuracy Check: {is_close}")
    print("="*50 + "\n")
    

if __name__ == "__main__":
    accuracy_test()