import torch
import math
from .tuner import jit_tuner

includes = ('"rope/rope.cuh"', )

template = """
fray::fused_rope<{HEAD_DIM}>(
    Q, K, cos_table, sin_table, cache_offsets, 
    num_tokens, num_heads, num_kv_heads, stream);
"""

def fused_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offsets: torch.Tensor = None) -> None:
    """
    Fused RoPE Kernel.
    q: [num_tokens, num_heads, head_dim]
    k: [num_tokens, num_kv_heads, head_dim]
    cos/sin: [max_seq_len, head_dim // 2]
    offsets: [num_tokens] 存储每个 token 的 pos
    """
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    
    # 连续性检查
    q = q.contiguous()
    k = k.contiguous()
    
    # 如果没有提供 offsets，默认认为是连续序列 [0, 1, 2...N-1]
    if offsets is None:
        offsets = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
    else:
        offsets = offsets.to(torch.int32)

    assert q.dtype == torch.half and k.dtype == torch.half
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32
    assert head_dim % 8 == 0, "HEAD_DIM 必须是 8 的倍数以支持 uint4 访存"

    stream = torch.cuda.current_stream()

    global includes, template

    args = (q, k, cos, sin, offsets, num_tokens, num_heads, num_kv_heads, stream)
    
    runtime = jit_tuner.compile_and_tune(
        name='fused_rope',
        keys={'HEAD_DIM': head_dim},
        space=(),
        includes=includes,
        arg_defs=(
            ('Q', torch.half),
            ('K', torch.half),
            ('cos_table', torch.float32),
            ('sin_table', torch.float32),
            ('cache_offsets', torch.int32),
            ('num_tokens', int),
            ('num_heads', int),
            ('num_kv_heads', int),
            ('stream', torch.cuda.Stream)
        ),
        template=template,
        args=args
    )

    runtime(*args)