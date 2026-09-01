import torch
import triton
import triton.language as tl


@triton.jit
def _rope_full_grouped_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    q_out_ptr,
    k_out_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)

    half_dim = head_dim // 2
    head_offs = head_block_id * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offs = tl.arange(0, BLOCK_HALF_D)
    dim_mask = dim_offs < half_dim

    position = tl.load(positions_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + dim_offs, mask=dim_mask, other=1.0)
    sin = tl.load(sin_ptr + position * half_dim + dim_offs, mask=dim_mask, other=0.0)
    cos = cos[None, :].to(tl.float32)
    sin = sin[None, :].to(tl.float32)

    if IS_NEOX_STYLE:
        x0_offset = dim_offs
        x1_offset = half_dim + dim_offs
    else:
        x0_offset = dim_offs * 2
        x1_offset = dim_offs * 2 + 1

    is_q = head_offs < num_q_heads
    q_head = tl.where(is_q, head_offs, 0)
    q_mask = is_q[:, None] & dim_mask[None, :]
    q_base = token_id * num_q_heads * head_dim + q_head[:, None] * head_dim
    q_x0 = tl.load(q_ptr + q_base + x0_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_x1 = tl.load(q_ptr + q_base + x1_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_y0 = q_x0 * cos - q_x1 * sin
    q_y1 = q_x0 * sin + q_x1 * cos
    tl.store(q_out_ptr + q_base + x0_offset[None, :], q_y0, mask=q_mask)
    tl.store(q_out_ptr + q_base + x1_offset[None, :], q_y1, mask=q_mask)

    is_k = (head_offs >= num_q_heads) & (head_offs < num_q_heads + num_kv_heads)
    k_head = tl.where(is_k, head_offs - num_q_heads, 0)
    k_mask = is_k[:, None] & dim_mask[None, :]
    k_base = token_id * num_kv_heads * head_dim + k_head[:, None] * head_dim
    k_x0 = tl.load(k_ptr + k_base + x0_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_x1 = tl.load(k_ptr + k_base + x1_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_y0 = k_x0 * cos - k_x1 * sin
    k_y1 = k_x0 * sin + k_x1 * cos
    tl.store(k_out_ptr + k_base + x0_offset[None, :], k_y0, mask=k_mask)
    tl.store(k_out_ptr + k_base + x1_offset[None, :], k_y1, mask=k_mask)


@triton.jit
def _rope_full_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    q_out_ptr,
    k_out_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_HALF_D)
    mask = offs < half_dim

    position = tl.load(positions_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + offs, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * half_dim + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    if head_id < num_q_heads:
        base = token_id * num_q_heads * head_dim + head_id * head_dim
        in_ptr = q_ptr + base
        out_ptr = q_out_ptr + base
    else:
        kv_head_id = head_id - num_q_heads
        base = token_id * num_kv_heads * head_dim + kv_head_id * head_dim
        in_ptr = k_ptr + base
        out_ptr = k_out_ptr + base

    if IS_NEOX_STYLE:
        x0_offset = offs
        x1_offset = half_dim + offs
    else:
        x0_offset = offs * 2
        x1_offset = offs * 2 + 1

    x0 = tl.load(in_ptr + x0_offset, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in_ptr + x1_offset, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    tl.store(out_ptr + x0_offset, y0, mask=mask)
    tl.store(out_ptr + x1_offset, y1, mask=mask)


@triton.jit
def _rope_full_k_cache_grouped_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    cache_positions_ptr,
    q_out_ptr,
    k_cache_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)

    half_dim = head_dim // 2
    head_offs = head_block_id * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offs = tl.arange(0, BLOCK_HALF_D)
    dim_mask = dim_offs < half_dim

    position = tl.load(positions_ptr + token_id)
    cache_position = tl.load(cache_positions_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + dim_offs, mask=dim_mask, other=1.0)
    sin = tl.load(sin_ptr + position * half_dim + dim_offs, mask=dim_mask, other=0.0)
    cos = cos[None, :].to(tl.float32)
    sin = sin[None, :].to(tl.float32)

    if IS_NEOX_STYLE:
        x0_offset = dim_offs
        x1_offset = half_dim + dim_offs
    else:
        x0_offset = dim_offs * 2
        x1_offset = dim_offs * 2 + 1

    is_q = head_offs < num_q_heads
    q_head = tl.where(is_q, head_offs, 0)
    q_mask = is_q[:, None] & dim_mask[None, :]
    q_base = token_id * num_q_heads * head_dim + q_head[:, None] * head_dim
    q_x0 = tl.load(q_ptr + q_base + x0_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_x1 = tl.load(q_ptr + q_base + x1_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_y0 = q_x0 * cos - q_x1 * sin
    q_y1 = q_x0 * sin + q_x1 * cos
    tl.store(q_out_ptr + q_base + x0_offset[None, :], q_y0, mask=q_mask)
    tl.store(q_out_ptr + q_base + x1_offset[None, :], q_y1, mask=q_mask)

    is_k = (head_offs >= num_q_heads) & (head_offs < num_q_heads + num_kv_heads)
    k_head = tl.where(is_k, head_offs - num_q_heads, 0)
    k_mask = is_k[:, None] & dim_mask[None, :]
    k_base = token_id * num_kv_heads * head_dim + k_head[:, None] * head_dim
    k_cache_base = cache_position * num_kv_heads * head_dim + k_head[:, None] * head_dim
    k_x0 = tl.load(k_ptr + k_base + x0_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_x1 = tl.load(k_ptr + k_base + x1_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_y0 = k_x0 * cos - k_x1 * sin
    k_y1 = k_x0 * sin + k_x1 * cos
    tl.store(k_cache_ptr + k_cache_base + x0_offset[None, :], k_y0, mask=k_mask)
    tl.store(k_cache_ptr + k_cache_base + x1_offset[None, :], k_y1, mask=k_mask)


@triton.jit
def _rope_full_paged_k_cache_grouped_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    page_indices_ptr,
    page_offsets_ptr,
    q_out_ptr,
    k_cache_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block_id = tl.program_id(1)

    half_dim = head_dim // 2
    head_offs = head_block_id * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offs = tl.arange(0, BLOCK_HALF_D)
    dim_mask = dim_offs < half_dim

    position = tl.load(positions_ptr + token_id)
    page_idx = tl.load(page_indices_ptr + token_id)
    page_offset = tl.load(page_offsets_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + dim_offs, mask=dim_mask, other=1.0)
    sin = tl.load(sin_ptr + position * half_dim + dim_offs, mask=dim_mask, other=0.0)
    cos = cos[None, :].to(tl.float32)
    sin = sin[None, :].to(tl.float32)

    if IS_NEOX_STYLE:
        x0_offset = dim_offs
        x1_offset = half_dim + dim_offs
    else:
        x0_offset = dim_offs * 2
        x1_offset = dim_offs * 2 + 1

    is_q = head_offs < num_q_heads
    q_head = tl.where(is_q, head_offs, 0)
    q_mask = is_q[:, None] & dim_mask[None, :]
    q_base = token_id * num_q_heads * head_dim + q_head[:, None] * head_dim
    q_x0 = tl.load(q_ptr + q_base + x0_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_x1 = tl.load(q_ptr + q_base + x1_offset[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q_y0 = q_x0 * cos - q_x1 * sin
    q_y1 = q_x0 * sin + q_x1 * cos
    tl.store(q_out_ptr + q_base + x0_offset[None, :], q_y0, mask=q_mask)
    tl.store(q_out_ptr + q_base + x1_offset[None, :], q_y1, mask=q_mask)

    is_k = (head_offs >= num_q_heads) & (head_offs < num_q_heads + num_kv_heads)
    k_head = tl.where(is_k, head_offs - num_q_heads, 0)
    k_mask = is_k[:, None] & dim_mask[None, :]
    k_base = token_id * num_kv_heads * head_dim + k_head[:, None] * head_dim
    k_cache_base = (
        page_idx * page_size + page_offset
    ) * num_kv_heads * head_dim + k_head[:, None] * head_dim
    k_x0 = tl.load(k_ptr + k_base + x0_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_x1 = tl.load(k_ptr + k_base + x1_offset[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_y0 = k_x0 * cos - k_x1 * sin
    k_y1 = k_x0 * sin + k_x1 * cos
    tl.store(k_cache_ptr + k_cache_base + x0_offset[None, :], k_y0, mask=k_mask)
    tl.store(k_cache_ptr + k_cache_base + x1_offset[None, :], k_y1, mask=k_mask)


@triton.jit
def _rope_partial_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    q_out_ptr,
    k_out_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_TAIL_D: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    COPY_TAIL: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    half_dim = rotary_dim // 2
    offs = tl.arange(0, BLOCK_HALF_D)
    mask = offs < half_dim

    position = tl.load(positions_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + offs, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * half_dim + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    if head_id < num_q_heads:
        base = token_id * num_q_heads * head_dim + head_id * head_dim
        in_ptr = q_ptr + base
        out_ptr = q_out_ptr + base
    else:
        kv_head_id = head_id - num_q_heads
        base = token_id * num_kv_heads * head_dim + kv_head_id * head_dim
        in_ptr = k_ptr + base
        out_ptr = k_out_ptr + base

    if IS_NEOX_STYLE:
        x0_offset = offs
        x1_offset = half_dim + offs
    else:
        x0_offset = offs * 2
        x1_offset = offs * 2 + 1

    x0 = tl.load(in_ptr + x0_offset, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in_ptr + x1_offset, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    tl.store(out_ptr + x0_offset, y0, mask=mask)
    tl.store(out_ptr + x1_offset, y1, mask=mask)

    if COPY_TAIL:
        tail_dim = head_dim - rotary_dim
        tail_offs = tl.arange(0, BLOCK_TAIL_D)
        tail_mask = tail_offs < tail_dim
        tail = tl.load(in_ptr + rotary_dim + tail_offs, mask=tail_mask, other=0.0)
        tl.store(out_ptr + rotary_dim + tail_offs, tail, mask=tail_mask)


@triton.jit
def _rope_k_cache_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    cache_positions_ptr,
    q_out_ptr,
    k_cache_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_TAIL_D: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    COPY_Q_TAIL: tl.constexpr,
    COPY_K_TAIL: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    half_dim = rotary_dim // 2
    offs = tl.arange(0, BLOCK_HALF_D)
    mask = offs < half_dim

    position = tl.load(positions_ptr + token_id)
    cache_position = tl.load(cache_positions_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + offs, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * half_dim + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    if IS_NEOX_STYLE:
        x0_offset = offs
        x1_offset = half_dim + offs
    else:
        x0_offset = offs * 2
        x1_offset = offs * 2 + 1

    if head_id < num_q_heads:
        base = token_id * num_q_heads * head_dim + head_id * head_dim
        in_ptr = q_ptr + base
        out_ptr = q_out_ptr + base
    else:
        kv_head_id = head_id - num_q_heads
        base = token_id * num_kv_heads * head_dim + kv_head_id * head_dim
        cache_base = cache_position * num_kv_heads * head_dim + kv_head_id * head_dim
        in_ptr = k_ptr + base
        out_ptr = k_cache_ptr + cache_base

    x0 = tl.load(in_ptr + x0_offset, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in_ptr + x1_offset, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    tl.store(out_ptr + x0_offset, y0, mask=mask)
    tl.store(out_ptr + x1_offset, y1, mask=mask)

    if COPY_Q_TAIL:
        if head_id < num_q_heads:
            tail_dim = head_dim - rotary_dim
            tail_offs = tl.arange(0, BLOCK_TAIL_D)
            tail_mask = tail_offs < tail_dim
            tail = tl.load(in_ptr + rotary_dim + tail_offs, mask=tail_mask, other=0.0)
            tl.store(out_ptr + rotary_dim + tail_offs, tail, mask=tail_mask)

    if COPY_K_TAIL:
        if head_id >= num_q_heads:
            tail_dim = head_dim - rotary_dim
            tail_offs = tl.arange(0, BLOCK_TAIL_D)
            tail_mask = tail_offs < tail_dim
            tail = tl.load(in_ptr + rotary_dim + tail_offs, mask=tail_mask, other=0.0)
            tl.store(out_ptr + rotary_dim + tail_offs, tail, mask=tail_mask)


@triton.jit
def _rope_paged_k_cache_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    page_indices_ptr,
    page_offsets_ptr,
    q_out_ptr,
    k_cache_ptr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_TAIL_D: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
    COPY_Q_TAIL: tl.constexpr,
    COPY_K_TAIL: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    half_dim = rotary_dim // 2
    offs = tl.arange(0, BLOCK_HALF_D)
    mask = offs < half_dim

    position = tl.load(positions_ptr + token_id)
    page_idx = tl.load(page_indices_ptr + token_id)
    page_offset = tl.load(page_offsets_ptr + token_id)
    cos = tl.load(cos_ptr + position * half_dim + offs, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * half_dim + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    if IS_NEOX_STYLE:
        x0_offset = offs
        x1_offset = half_dim + offs
    else:
        x0_offset = offs * 2
        x1_offset = offs * 2 + 1

    if head_id < num_q_heads:
        base = token_id * num_q_heads * head_dim + head_id * head_dim
        in_ptr = q_ptr + base
        out_ptr = q_out_ptr + base
    else:
        kv_head_id = head_id - num_q_heads
        base = token_id * num_kv_heads * head_dim + kv_head_id * head_dim
        cache_base = (
            page_idx * page_size + page_offset
        ) * num_kv_heads * head_dim + kv_head_id * head_dim
        in_ptr = k_ptr + base
        out_ptr = k_cache_ptr + cache_base

    x0 = tl.load(in_ptr + x0_offset, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in_ptr + x1_offset, mask=mask, other=0.0).to(tl.float32)

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    tl.store(out_ptr + x0_offset, y0, mask=mask)
    tl.store(out_ptr + x1_offset, y1, mask=mask)

    if COPY_Q_TAIL:
        if head_id < num_q_heads:
            tail_dim = head_dim - rotary_dim
            tail_offs = tl.arange(0, BLOCK_TAIL_D)
            tail_mask = tail_offs < tail_dim
            tail = tl.load(in_ptr + rotary_dim + tail_offs, mask=tail_mask, other=0.0)
            tl.store(out_ptr + rotary_dim + tail_offs, tail, mask=tail_mask)

    if COPY_K_TAIL:
        if head_id >= num_q_heads:
            tail_dim = head_dim - rotary_dim
            tail_offs = tl.arange(0, BLOCK_TAIL_D)
            tail_mask = tail_offs < tail_dim
            tail = tl.load(in_ptr + rotary_dim + tail_offs, mask=tail_mask, other=0.0)
            tl.store(out_ptr + rotary_dim + tail_offs, tail, mask=tail_mask)


def _get_num_warps(block_half_d: int) -> int:
    if block_half_d <= 64:
        return 4
    return 8


def _get_block_h(block_half_d: int) -> int:
    if block_half_d <= 64:
        return 4
    return 2


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda, (
        "q, k, cos, sin must be CUDA tensors"
    )

    assert q.dim() == 3 and k.dim() == 3, "q and k must be 3D tensors"

    num_tokens, num_q_heads, head_dim = q.shape
    k_num_tokens, num_kv_heads, k_head_dim = k.shape

    if q_out is None:
        q_out = torch.empty_like(q)
    if k_out is None:
        k_out = torch.empty_like(k)
    if positions is None:
        positions = torch.arange(num_tokens, device=q.device, dtype=torch.int32)

    assert q_out.is_cuda and k_out.is_cuda, "q_out and k_out must be CUDA tensors"
    assert positions.is_cuda, "positions must be a CUDA tensor"
    assert q_out.shape == q.shape and k_out.shape == k.shape, (
        "q_out and k_out must match q and k shapes"
    )

    assert k_num_tokens == num_tokens, "q and k must have the same token dimension"
    assert k_head_dim == head_dim, "q and k must have the same head_dim"
    assert head_dim % 2 == 0, "head_dim must be even"

    if rotary_dim is None:
        rotary_dim = head_dim
    assert rotary_dim > 0 and rotary_dim <= head_dim, (
        "rotary_dim must be in the range (0, head_dim]"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    half_dim = rotary_dim // 2
    assert cos.dim() == 2 and sin.dim() == 2, "cos and sin must be 2D tensors"
    assert cos.shape == sin.shape, "cos and sin must have the same shape"
    assert cos.shape[1] == half_dim, (
        f"cos/sin last dimension must be {half_dim}, got {cos.shape[1]}"
    )
    assert positions.dim() == 1 and positions.shape[0] == num_tokens, (
        "positions must be a 1D tensor with one position per token"
    )

    assert q.dtype in (torch.float16, torch.bfloat16), "q dtype must be fp16 or bf16"
    assert k.dtype == q.dtype and q_out.dtype == q.dtype and k_out.dtype == q.dtype, (
        "q, k, q_out, k_out must have the same dtype"
    )
    assert cos.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "cos dtype must be fp16, bf16, or fp32"
    )
    assert sin.dtype == cos.dtype, "sin dtype must match cos dtype"
    assert positions.dtype in (torch.int32, torch.int64), (
        "positions dtype must be int32 or int64"
    )

    assert q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous"
    assert q_out.is_contiguous() and k_out.is_contiguous(), (
        "q_out and k_out must be contiguous"
    )
    assert cos.is_contiguous() and sin.is_contiguous(), "cos and sin must be contiguous"
    assert positions.is_contiguous(), "positions must be contiguous"

    BLOCK_HALF_D = triton.next_power_of_2(half_dim)
    grid = (num_tokens, num_q_heads + num_kv_heads)

    if rotary_dim == head_dim:
        BLOCK_H = _get_block_h(BLOCK_HALF_D)
        full_grid = (num_tokens, triton.cdiv(num_q_heads + num_kv_heads, BLOCK_H))
        _rope_full_grouped_kernel[full_grid](
            q,
            k,
            cos,
            sin,
            positions,
            q_out,
            k_out,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_H=BLOCK_H,
            IS_NEOX_STYLE=is_neox_style,
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    else:
        tail_dim = head_dim - rotary_dim
        BLOCK_TAIL_D = triton.next_power_of_2(tail_dim)
        _rope_partial_kernel[grid](
            q,
            k,
            cos,
            sin,
            positions,
            q_out,
            k_out,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_TAIL_D=BLOCK_TAIL_D,
            IS_NEOX_STYLE=is_neox_style,
            COPY_TAIL=not (q_out is q and k_out is k),
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    return q_out, k_out


def rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return rope(
        q,
        k,
        cos,
        sin,
        positions=positions,
        q_out=q,
        k_out=k,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def rope_with_k_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None,
    k_cache: torch.Tensor,
    cache_positions: torch.Tensor | None = None,
    q_out: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda, (
        "q, k, cos, sin must be CUDA tensors"
    )
    assert k_cache.is_cuda, "k_cache must be a CUDA tensor"
    assert q.dim() == 3 and k.dim() == 3, "q and k must be 3D tensors"
    assert k_cache.dim() == 3, "k_cache must be a 3D tensor"

    num_tokens, num_q_heads, head_dim = q.shape
    k_num_tokens, num_kv_heads, k_head_dim = k.shape

    if q_out is None:
        q_out = torch.empty_like(q)
    if positions is None:
        positions = torch.arange(num_tokens, device=q.device, dtype=torch.int32)
    if cache_positions is None:
        cache_positions = positions

    assert q_out.is_cuda, "q_out must be a CUDA tensor"
    assert positions.is_cuda and cache_positions.is_cuda, (
        "positions and cache_positions must be CUDA tensors"
    )
    assert q_out.shape == q.shape, "q_out must match q shape"
    assert k_cache.shape[1:] == k.shape[1:], (
        "k_cache must have shape [cache_tokens, num_kv_heads, head_dim]"
    )

    assert k_num_tokens == num_tokens, "q and k must have the same token dimension"
    assert k_head_dim == head_dim, "q and k must have the same head_dim"
    assert head_dim % 2 == 0, "head_dim must be even"

    if rotary_dim is None:
        rotary_dim = head_dim
    assert rotary_dim > 0 and rotary_dim <= head_dim, (
        "rotary_dim must be in the range (0, head_dim]"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    half_dim = rotary_dim // 2
    assert cos.dim() == 2 and sin.dim() == 2, "cos and sin must be 2D tensors"
    assert cos.shape == sin.shape, "cos and sin must have the same shape"
    assert cos.shape[1] == half_dim, (
        f"cos/sin last dimension must be {half_dim}, got {cos.shape[1]}"
    )
    assert positions.dim() == 1 and positions.shape[0] == num_tokens, (
        "positions must be a 1D tensor with one position per token"
    )
    assert cache_positions.dim() == 1 and cache_positions.shape[0] == num_tokens, (
        "cache_positions must be a 1D tensor with one cache slot per token"
    )

    assert q.dtype in (torch.float16, torch.bfloat16), "q dtype must be fp16 or bf16"
    assert k.dtype == q.dtype and q_out.dtype == q.dtype and k_cache.dtype == q.dtype, (
        "q, k, q_out, k_cache must have the same dtype"
    )
    assert cos.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "cos dtype must be fp16, bf16, or fp32"
    )
    assert sin.dtype == cos.dtype, "sin dtype must match cos dtype"
    assert positions.dtype in (torch.int32, torch.int64), (
        "positions dtype must be int32 or int64"
    )
    assert cache_positions.dtype in (torch.int32, torch.int64), (
        "cache_positions dtype must be int32 or int64"
    )

    assert q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous"
    assert q_out.is_contiguous(), "q_out must be contiguous"
    assert k_cache.is_contiguous(), "k_cache must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous(), "cos and sin must be contiguous"
    assert positions.is_contiguous(), "positions must be contiguous"
    assert cache_positions.is_contiguous(), "cache_positions must be contiguous"

    BLOCK_HALF_D = triton.next_power_of_2(half_dim)
    tail_dim = head_dim - rotary_dim
    BLOCK_TAIL_D = max(1, triton.next_power_of_2(tail_dim))
    grid = (num_tokens, num_q_heads + num_kv_heads)

    if rotary_dim == head_dim:
        BLOCK_H = _get_block_h(BLOCK_HALF_D)
        full_grid = (num_tokens, triton.cdiv(num_q_heads + num_kv_heads, BLOCK_H))
        _rope_full_k_cache_grouped_kernel[full_grid](
            q,
            k,
            cos,
            sin,
            positions,
            cache_positions,
            q_out,
            k_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_H=BLOCK_H,
            IS_NEOX_STYLE=is_neox_style,
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    else:
        _rope_k_cache_kernel[grid](
            q,
            k,
            cos,
            sin,
            positions,
            cache_positions,
            q_out,
            k_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_TAIL_D=BLOCK_TAIL_D,
            IS_NEOX_STYLE=is_neox_style,
            COPY_Q_TAIL=rotary_dim < head_dim and q_out is not q,
            COPY_K_TAIL=rotary_dim < head_dim,
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    return q_out, k_cache


def rope_with_paged_k_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None,
    k_cache: torch.Tensor,
    page_indices: torch.Tensor,
    page_offsets: torch.Tensor,
    q_out: torch.Tensor | None = None,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda, (
        "q, k, cos, sin must be CUDA tensors"
    )
    assert k_cache.is_cuda, "k_cache must be a CUDA tensor"
    assert page_indices.is_cuda and page_offsets.is_cuda, (
        "page_indices and page_offsets must be CUDA tensors"
    )
    assert q.dim() == 3 and k.dim() == 3, "q and k must be 3D tensors"
    assert k_cache.dim() == 4, (
        "k_cache must be a 4D tensor with shape "
        "[num_pages, page_size, num_kv_heads, head_dim]"
    )

    num_tokens, num_q_heads, head_dim = q.shape
    k_num_tokens, num_kv_heads, k_head_dim = k.shape
    _, page_size, cache_num_kv_heads, cache_head_dim = k_cache.shape

    if q_out is None:
        q_out = torch.empty_like(q)
    if positions is None:
        positions = torch.arange(num_tokens, device=q.device, dtype=torch.int32)

    assert q_out.is_cuda, "q_out must be a CUDA tensor"
    assert positions.is_cuda, "positions must be a CUDA tensor"
    assert q_out.shape == q.shape, "q_out must match q shape"
    assert cache_num_kv_heads == num_kv_heads and cache_head_dim == head_dim, (
        "k_cache must have shape [num_pages, page_size, num_kv_heads, head_dim]"
    )

    assert k_num_tokens == num_tokens, "q and k must have the same token dimension"
    assert k_head_dim == head_dim, "q and k must have the same head_dim"
    assert head_dim % 2 == 0, "head_dim must be even"

    if rotary_dim is None:
        rotary_dim = head_dim
    assert rotary_dim > 0 and rotary_dim <= head_dim, (
        "rotary_dim must be in the range (0, head_dim]"
    )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    half_dim = rotary_dim // 2
    assert cos.dim() == 2 and sin.dim() == 2, "cos and sin must be 2D tensors"
    assert cos.shape == sin.shape, "cos and sin must have the same shape"
    assert cos.shape[1] == half_dim, (
        f"cos/sin last dimension must be {half_dim}, got {cos.shape[1]}"
    )
    assert positions.dim() == 1 and positions.shape[0] == num_tokens, (
        "positions must be a 1D tensor with one position per token"
    )
    assert page_indices.dim() == 1 and page_indices.shape[0] == num_tokens, (
        "page_indices must be a 1D tensor with one page index per token"
    )
    assert page_offsets.dim() == 1 and page_offsets.shape[0] == num_tokens, (
        "page_offsets must be a 1D tensor with one page offset per token"
    )

    assert q.dtype in (torch.float16, torch.bfloat16), "q dtype must be fp16 or bf16"
    assert k.dtype == q.dtype and q_out.dtype == q.dtype and k_cache.dtype == q.dtype, (
        "q, k, q_out, k_cache must have the same dtype"
    )
    assert cos.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "cos dtype must be fp16, bf16, or fp32"
    )
    assert sin.dtype == cos.dtype, "sin dtype must match cos dtype"
    assert positions.dtype in (torch.int32, torch.int64), (
        "positions dtype must be int32 or int64"
    )
    assert page_indices.dtype in (torch.int32, torch.int64), (
        "page_indices dtype must be int32 or int64"
    )
    assert page_offsets.dtype in (torch.int32, torch.int64), (
        "page_offsets dtype must be int32 or int64"
    )

    assert q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous"
    assert q_out.is_contiguous(), "q_out must be contiguous"
    assert k_cache.is_contiguous(), "k_cache must be contiguous"
    assert cos.is_contiguous() and sin.is_contiguous(), "cos and sin must be contiguous"
    assert positions.is_contiguous(), "positions must be contiguous"
    assert page_indices.is_contiguous(), "page_indices must be contiguous"
    assert page_offsets.is_contiguous(), "page_offsets must be contiguous"

    BLOCK_HALF_D = triton.next_power_of_2(half_dim)
    tail_dim = head_dim - rotary_dim
    BLOCK_TAIL_D = max(1, triton.next_power_of_2(tail_dim))
    grid = (num_tokens, num_q_heads + num_kv_heads)

    if rotary_dim == head_dim:
        BLOCK_H = _get_block_h(BLOCK_HALF_D)
        full_grid = (num_tokens, triton.cdiv(num_q_heads + num_kv_heads, BLOCK_H))
        _rope_full_paged_k_cache_grouped_kernel[full_grid](
            q,
            k,
            cos,
            sin,
            positions,
            page_indices,
            page_offsets,
            q_out,
            k_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_H=BLOCK_H,
            IS_NEOX_STYLE=is_neox_style,
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    else:
        _rope_paged_k_cache_kernel[grid](
            q,
            k,
            cos,
            sin,
            positions,
            page_indices,
            page_offsets,
            q_out,
            k_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            page_size=page_size,
            BLOCK_HALF_D=BLOCK_HALF_D,
            BLOCK_TAIL_D=BLOCK_TAIL_D,
            IS_NEOX_STYLE=is_neox_style,
            COPY_Q_TAIL=rotary_dim < head_dim and q_out is not q,
            COPY_K_TAIL=rotary_dim < head_dim,
            num_warps=_get_num_warps(BLOCK_HALF_D),
        )
    return q_out, k_cache
