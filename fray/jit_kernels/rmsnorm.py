import torch
import math
from .tuner import jit_tuner

includes = ('"norm/fused_rmsnorm.cuh"', )
template = """
fray::fused_rmsnorm(output, residual, input, weight, epsilon, num_tokens, hidden_dim, stream);
"""

def fused_rmsnorm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    num_tokens = input.numel() // input.shape[-1]
    hidden_dim = input.shape[-1]

    if not input.is_contiguous(): input = input.contiguous()
    if not residual.is_contiguous(): residual = residual.contiguous()
    if not weight.is_contiguous(): weight = weight.contiguous()

    output = torch.empty_like(input)

    assert hidden_dim % 8 == 0, "hidden_dim must be a multiple of 8 for 128-bit vectorization"
    assert input.dtype == torch.half and residual.dtype == torch.half and weight.dtype == torch.half

    stream = torch.cuda.current_stream()

    global includes, template

    args = (output, residual, input, weight, epsilon, num_tokens, hidden_dim, stream)

    runtime = jit_tuner.compile_and_tune(
        name='fused_rmsnorm',
        keys={'D': hidden_dim}, # 根据 hidden_dim 缓存编译结果
        space=(),
        includes=includes,
        arg_defs=(
            ('output', torch.half), 
            ('residual', torch.half), 
            ('input', torch.half), 
            ('weight', torch.half), 
            ('epsilon', float), 
            ('num_tokens', int), 
            ('hidden_dim', int), 
            ('stream', torch.cuda.Stream)
        ),
        template=template,
        args=args
    )

    runtime(*args)
    return output