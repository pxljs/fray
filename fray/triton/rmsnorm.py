import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < n_cols

    row_start = row_id * n_cols

    x = tl.load(x_ptr + row_start + offs_n, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)

    # 计算 RMS
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    out = x * rstd * weight

    tl.store(output_ptr + row_start + offs_n, out, mask=mask)

def _get_num_warps(block_n: int) -> int:
    if block_n <= 1024:
        return 4
    elif block_n <= 2048:
        return 8
    else:
        return 8

def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
):
    assert x.is_cuda and weight.is_cuda and output.is_cuda, \
        "All tensors must be CUDA tensors"

    assert x.is_contiguous() and weight.is_contiguous() and output.is_contiguous(), \
        "All tensors must be contiguous"

    assert x.shape == output.shape, \
        "x and output must have the same shape"

    assert weight.dim() == 1, \
        "weight must be a 1D tensor"

    assert x.shape[-1] == weight.shape[0], \
        "weight shape must match the last dimension of x"

    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    # Triton 的 tl.arange(0, BLOCK_N) 通常要求 BLOCK_N 是 2 的幂，
    # 所以这里把 hidden size 向上取到 2 的幂。
    BLOCK_N = triton.next_power_of_2(n_cols)

    num_warps = _get_num_warps(BLOCK_N)

    grid = (n_rows,)

    _rmsnorm_kernel[grid](
        x,
        weight,
        output,
        n_cols,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )