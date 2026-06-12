import torch
import math
from .tuner import jit_tuner

includes = ('"flash_mla/flash_mla.cuh"', )

template = """
fray::flash_mla_paged_decoding<{R}, {C}, {D_nope}, {D_rope}, {K}>(
    reinterpret_cast<const cute::half_t*>(Q),
    reinterpret_cast<const cute::half_t*>(KV_cache),
    reinterpret_cast<const int*>(block_table),
    reinterpret_cast<const int*>(seq_lens),
    reinterpret_cast<cute::half_t*>(O_intermediate),
    batch_size, num_heads, max_blocks_per_seq, scale, stream
);
"""

def flash_mla(
    Q: torch.Tensor, 
    KV_cache: torch.Tensor, 
    block_table: torch.Tensor, 
    seq_lens: torch.Tensor, 
    O_intermediate: torch.Tensor, 
    scale: float = None
) -> None:
    # Q:              [B, 1, H_q, 576] (Q 序列长固定为 1)
    # KV_cache:       [Max_Pages, Block_Size=64, 576]
    # block_table:    [B, Max_Blocks_Per_Seq]
    # seq_lens:       [B]
    # O_intermediate: [B, H_q, 512] (在 512 维潜变量空间的中间输出)
    
    batch_size, q_len, num_heads, d_q = Q.shape
    max_blocks_per_seq = block_table.shape[1]

    assert q_len == 1, "flash_mla 仅支持 decoding 阶段,Q 的序列长度必须为 1"
    assert d_q == 576, "Q 最后一维必须是 576"
    assert KV_cache.shape[1] == 64 and KV_cache.shape[2] == 576, "KV_cache 形状必须是 [Max_Pages, 64, 576]"
    assert O_intermediate.shape == (batch_size, num_heads, 512), "O_intermediate 形状必须是 [B, H_q, 512]"
    assert O_intermediate.is_contiguous(), "O_intermediate 必须是 contiguous 输出张量"
    assert num_heads % 8 == 0, "num_heads 必须是 8 的倍数"
    
    if scale is None:
        scale = 1.0 / math.sqrt(576.0)
        
    R = 16 if num_heads % 16 == 0 else 8 # kBlockM: 每个 Block 处理的 Query 头数
    C = 64       # kBlockN: 每个 Block 迭代的序列分块长度
    D_nope = 512 # kDimNope: 隐空间通道长度
    D_rope = 64  # kDimRope: 旋转编码通道长度
    K = 128      # kBlockK: 隐空间的 Tiling 宽度（512 分 4 次加载）
    tuning_space = ({'R': 8}, {'R': 16}) if num_heads % 16 == 0 else ({'R': 8},)
    
    # 输入可安全复制为连续内存；输出必须保持调用方传入的原 tensor。
    Q, KV_cache, block_table, seq_lens = (
        Q.contiguous(), 
        KV_cache.contiguous(), 
        block_table.contiguous(), 
        seq_lens.contiguous()
    )
    
    stream = torch.cuda.current_stream()
    
    global includes, template
    
    args = (
        Q, KV_cache, block_table, seq_lens, O_intermediate, 
        batch_size, num_heads, max_blocks_per_seq, scale, stream
    )
    
    # 调用 jit_tuner 进行即时编译与参数调优
    runtime = jit_tuner.compile_and_tune(
        name='flash_mla_paged_decoding',
        keys={'R': R, 'C': C, 'D_nope': D_nope, 'D_rope': D_rope, 'K': K, 'H_q': num_heads}, 
        space=tuning_space,
        includes=includes,
        arg_defs=(
            ('Q', torch.half), ('KV_cache', torch.half), ('block_table', torch.int32), 
            ('seq_lens', torch.int32), ('O_intermediate', torch.half),
            ('batch_size', int), ('num_heads', int), ('max_blocks_per_seq', int), 
            ('scale', float), ('stream', torch.cuda.Stream)
        ),
        template=template,
        args=args
    )

    # 执行算子
    runtime(*args)
