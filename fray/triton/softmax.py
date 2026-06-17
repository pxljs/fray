import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    n_cols,
    q_len,
    scale,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    valid_mask = offs_n < n_cols

    row_start = row_id * n_cols

    x = tl.load(
        x_ptr + row_start + offs_n,
        mask=valid_mask,
        other=-float("inf"),
    ).to(tl.float32)

    x = x * scale

    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    if HAS_MASK:
        if MASK_IS_BOOL:
            # bool mask 语义：
            #   True  表示保留
            #   False 表示屏蔽
            m = tl.load(
                mask_ptr + row_start + offs_n,
                mask=valid_mask,
                other=0,
            )
            x = tl.where(m & valid_mask, x, -float("inf"))
        else:
            # additive mask 语义：
            #   0      表示保留
            #   -inf   表示屏蔽
            #   或者其他 bias 值
            m = tl.load(
                mask_ptr + row_start + offs_n,
                mask=valid_mask,
                other=-float("inf"),
            ).to(tl.float32)
            x = x + m

    if IS_CAUSAL:
        # 对最后两维 [..., q_len, n_cols] 做 causal mask。
        # row_id % q_len 得到当前 query row 在最后二维矩阵中的行号。
        q_row = row_id % q_len
        causal_mask = offs_n <= q_row
        x = tl.where(causal_mask & valid_mask, x, -float("inf"))

    row_max = tl.max(x, axis=0)

    # 防止整行都被 mask 掉时出现 -inf - -inf = NaN
    row_max = tl.where(row_max == -float("inf"), 0.0, row_max)

    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)

    denominator = tl.where(denominator == 0.0, 1.0, denominator)

    out = numerator / denominator

    tl.store(
        output_ptr + row_start + offs_n,
        out,
        mask=valid_mask,
    )


def _get_num_warps(block_n: int) -> int:
    if block_n <= 1024:
        return 4
    elif block_n <= 2048:
        return 8
    else:
        return 8


def softmax(
    x: torch.Tensor,
    output: torch.Tensor,
    scale: float = 1.0,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    assert x.is_cuda and output.is_cuda, \
        "x and output must be CUDA tensors"

    assert x.is_contiguous() and output.is_contiguous(), \
        "x and output must be contiguous"

    assert x.shape == output.shape, \
        "x and output must have the same shape"

    assert x.dim() >= 2, \
        "x must have at least 2 dimensions"

    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    q_len = x.shape[-2]

    HAS_MASK = mask is not None
    MASK_IS_BOOL = False

    if HAS_MASK:
        assert mask.is_cuda, \
            "mask must be a CUDA tensor"

        assert mask.is_contiguous(), \
            "mask must be contiguous"

        assert mask.shape == x.shape, \
            "first version only supports mask with the same shape as x"

        if mask.dtype == torch.bool:
            MASK_IS_BOOL = True
        else:
            assert mask.dtype in (torch.float16, torch.bfloat16, torch.float32), \
                "mask must be bool, fp16, bf16, or fp32"

        mask_arg = mask
    else:
        # HAS_MASK=False 时 kernel 不会使用 mask_ptr。
        # 这里传 x 只是为了占位，避免传 None。
        mask_arg = x

    BLOCK_N = triton.next_power_of_2(n_cols)

    assert BLOCK_N <= 131072, \
        f"n_cols={n_cols} is too large for one-program-per-row softmax"

    num_warps = _get_num_warps(BLOCK_N)

    grid = (n_rows,)

    _softmax_kernel[grid](
        x,
        mask_arg,
        output,
        n_cols,
        q_len,
        scale,
        HAS_MASK=HAS_MASK,
        MASK_IS_BOOL=MASK_IS_BOOL,
        IS_CAUSAL=is_causal,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return output