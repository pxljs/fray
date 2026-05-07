import torch
from .tuner import jit_tuner

includes = ('"softmax/softmax.cuh"', )
template = """
// Templated args from Python JIT call
fray::softmax_c<float, {BLOCK_SIZE}>(X ,Y ,B, H, S, D, stride_b, stride_h, stride_s, stride_d);
"""

def softmax(x: torch.Tensor, y: torch.Tensor) -> None:
    B, H, S, D = x.shape
    stride_b = x.stride(0)
    stride_h = x.stride(1)
    stride_s = x.stride(2)
    stride_d = x.stride(3)
    assert x.dtype == torch.float32 and y.dtype == torch.float32

    # assert N % 4 == 0, "N 必须是 4 的倍数以支持 float4 向量化加载"

    global includes, template
    
    args = (x, y, B, H, S, D, stride_b, stride_h, stride_s, stride_d)

    # JIT compile and runtime tunner 
    runtime = jit_tuner.compile_and_tune(
        name='softmax',
        keys={'BLOCK_SIZE': 256},
        space = ({'BLOCK_SIZE': 128}, {'BLOCK_SIZE': 256}, {'BLOCK_SIZE': 512}, {'BLOCK_SIZE': 1024}),
        includes=includes,
        arg_defs=(('X', torch.float), ('Y', torch.float),
                  ('B', int), ('H', int), ('S', int), ('D', int), 
                  ('stride_b', int), ('stride_h', int),('stride_s', int), ('stride_d', int)),
        template=template,
        args=args
    )

    runtime(*args)


def accuracy_test():
    torch.manual_seed(42)
    
    # 模拟大模型中常见的尺寸[Batch, Heads, SeqLen, HeadDim]
    B, H, S, D = 2, 12, 4096, 128 
    
    print("\n" + "="*50)
    print(" 🧪 Test 1: 连续内存张量 (Contiguous Tensor)")
    print("="*50)
    # x 是内存连续的
    x = torch.randn(B, H, S, D, dtype=torch.float, device='cuda')
    y_fray = torch.empty_like(x)
    
    # 1. 运行自研算子
    softmax(x, y_fray)
    
    # 2. 运行 PyTorch 原生算子
    y_ref = torch.nn.functional.softmax(x, dim=-1)
    
    # 取一个元素切片打印对比
    print(f"Fray Output Sample: {y_fray[0, 0, 0, :4].tolist()}")
    print(f"Torch Ref Sample  : {y_ref[0, 0, 0, :4].tolist()}")
    
    # 3. 数值验证
    assert torch.allclose(y_fray, y_ref, rtol=1e-3, atol=1e-5), "连续内存 Softmax 精度错误!"
    print("✅ 连续内存测试通过 (Contiguous Test Passed)!\n")


    print("="*50)
    print(" 🧪 Test 2: 非连续内存张量 (Transposed / Non-Contiguous Tensor)")
    print("="*50)
    # 在大模型注意力机制中，经常发生 seq_len 和 heads 的维度交换
    # 这里的 x_trans 物理内存是不连续的，它的 stride_h 和 stride_s 被打乱了
    x_raw = torch.randn(B, S, H, D, dtype=torch.float, device='cuda')
    x_trans = x_raw.transpose(1, 2) # 形状变成了 [B, H, S, D]，但内存步长不连续
    
    y_trans_fray = torch.empty_like(x_trans)
    
    # 1. 运行自研算子 (CuTe 会自动通过 stride 解析正确的物理地址！)
    softmax(x_trans, y_trans_fray)
    
    # 2. 运行 PyTorch 原生算子
    y_trans_ref = torch.nn.functional.softmax(x_trans, dim=-1)
    
    # 3. 数值验证
    assert torch.allclose(y_trans_fray, y_trans_ref, rtol=1e-3, atol=1e-5), "非连续内存 Softmax 精度错误!"
    print("✅ 非连续内存测试通过 (Non-Contiguous Test Passed)!")

if __name__ == "__main__":
    accuracy_test()