import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .tuner import jit_tuner

includes = ('"flash_attn/flashattn_cute.cuh"', )
template = """
// Templated args from Python JIT call
fray::flash_attn_cute<{R},{C},{D}>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
"""

def flash_attn_cute(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor) -> None:
    batch_size, num_heads, seq_len, d = Q.shape
    
    if not Q.is_contiguous(): Q = Q.contiguous()
    if not K.is_contiguous(): K = K.contiguous()
    if not V.is_contiguous(): V = V.contiguous()
    if not Output.is_contiguous(): Output = Output.contiguous()
    
    assert Q.shape == K.shape == V.shape == Output.shape
    assert Q.dtype == K.dtype == V.dtype == Output.dtype == torch.half
    assert Q.device.type == "cuda"
    
    stream = torch.cuda.current_stream()

    global includes, template

    args = (Q, K, V, Output, batch_size, num_heads, seq_len, stream)
    runtime = jit_tuner.compile_and_tune(
        name='flash_attn_cute',
        keys={'R':128, 'C':32, 'D':d},
        space=({'R':64, 'C':32, 'D':d},
               {'R':128, 'C':32, 'D':d},
               {'R':64, 'C':128, 'D':d},
               {'R':64, 'C':64, 'D':d}),
        includes=includes,
        arg_defs=(('Q', torch.half), ('K', torch.half), ('V', torch.half), ('O', torch.half), ('batch_size', int), ('num_heads', int), ('seq_len', int), ('stream', torch.cuda.Stream)),
        template=template,
        args=args
    )

    runtime(*args)