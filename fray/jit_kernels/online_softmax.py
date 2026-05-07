import torch
from .tuner import jit_tuner

includes = ('"softmax/online_softmax.cuh"', )
template = """
// Templated args from Python JIT call
fray::online_softmax_c(X, Y, M, D);
"""

def online_softmax(x: torch.Tensor, y: torch.Tensor) -> None:
    M, D = x.shape
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    N = M * D
    assert N % 4 == 0, "N 必须是 4 的倍数以支持 float4 向量化加载"

    global includes, template
    
    args = (x, y, M, D)

    # JIT compile and runtime tunner 
    runtime = jit_tuner.compile_and_tune(
        name='online_softmax',
        keys={},
        space = (),
        includes=includes,
        arg_defs=(('X', torch.float), ('Y', torch.float), ('M', int), ('D', int)),
        template=template,
        args=args
    )

    runtime(*args)


def accuracy_test():
    torch.manual_seed(42)
    
    M, D = 16384, 512 
    
    x = torch.randn(M, D, dtype=torch.float, device='cuda')
    y_fray = torch.empty_like(x)
    
    online_softmax(x, y_fray)

    y_ref = torch.nn.functional.softmax(x, dim=-1)
    
    # 取一个元素切片打印对比
    print(f"Fray Output Sample: {y_fray[0, :4].tolist()}")
    print(f"Torch Ref Sample  : {y_ref[0, :4].tolist()}")
    
    # 数值验证
    assert torch.allclose(y_fray, y_ref, rtol=1e-3, atol=1e-5), "Online Softmax 精度错误!"
    print("✅ 测试通过 (Test Passed)!\n")

if __name__ == "__main__":
    accuracy_test()