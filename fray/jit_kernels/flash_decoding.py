import torch
import math
from .tuner import jit_tuner

includes = ('"flash_attn/flash_decoding.cuh"', )

template = """
fray::flash_decoding_cute<{G}, {BC}, {D}>(
    reinterpret_cast<cute::half_t*>(Q),
    reinterpret_cast<cute::half_t*>(K),
    reinterpret_cast<cute::half_t*>(V),
    reinterpret_cast<cute::half_t*>(O),
    workspace, 
    batch_size, num_heads, num_kv_heads, seq_len, stream
);
"""

def flash_decoding(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Output: torch.Tensor) -> None:
    batch_size, num_heads, _, d = Q.shape  # Q 序列长为 1
    seq_len = K.shape[2]
    num_kv_heads = K.shape[1]
    
    G = num_heads // num_kv_heads # 分组比
    Bc = 128
    num_blocks = math.ceil(seq_len / Bc)

    # O_tmp: [B, H, num_blocks, D] * 4 bytes
    # m_tmp: [B, H, num_blocks] * 4 bytes
    # l_tmp: [B, H, num_blocks] * 4 bytes
    workspace_size_bytes = (
        batch_size * num_heads * num_blocks * d * 4 + 
        batch_size * num_heads * num_blocks * 4 * 2
    )
    workspace = torch.empty(workspace_size_bytes, dtype=torch.uint8, device=Q.device)

    Q, K, V, Output = Q.contiguous(), K.contiguous(), V.contiguous(), Output.contiguous()
    stream = torch.cuda.current_stream()
    
    global includes, template

    args = (Q, K, V, Output, workspace, batch_size, num_heads, num_kv_heads, seq_len, stream)
    
    runtime = jit_tuner.compile_and_tune(
        name='flash_decoding',
        keys={'G': G, 'BC': Bc, 'D': d}, 
        space=(),
        includes=includes,
        arg_defs=(
            ('Q', torch.half), ('K', torch.half), ('V', torch.half), ('O', torch.half), 
            ('workspace', torch.uint8),
            ('batch_size', int), ('num_heads', int), ('num_kv_heads', int), 
            ('seq_len', int), ('stream', torch.cuda.Stream)
        ),
        template=template,
        args=args
    )

    runtime(*args)