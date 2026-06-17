import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=2,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group

    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    local_pid = pid % num_pid_in_group

    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )

        b = tl.load(
            b_ptrs,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offs_n,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(
        c_ptrs,
        acc,
        mask=c_mask,
    )


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    assert a.is_cuda and b.is_cuda and c.is_cuda, \
        "a, b, c must be CUDA tensors"

    assert a.dim() == 2 and b.dim() == 2 and c.dim() == 2, \
        "a, b, c must be 2D tensors"

    M, K = a.shape
    K_b, N = b.shape

    assert K == K_b, \
        f"shape mismatch: a.shape={a.shape}, b.shape={b.shape}"

    assert c.shape == (M, N), \
        f"c.shape must be ({M}, {N}), got {c.shape}"

    assert a.dtype in (torch.float16, torch.bfloat16), \
        "a dtype must be fp16 or bf16"

    assert b.dtype == a.dtype and c.dtype == a.dtype, \
        "a, b, c must have the same dtype"

    HAS_BIAS = bias is not None

    if HAS_BIAS:
        assert bias.is_cuda, \
            "bias must be a CUDA tensor"

        assert bias.dim() == 1, \
            "bias must be a 1D tensor"

        assert bias.shape[0] == N, \
            f"bias shape must be [{N}], got {bias.shape}"

        assert bias.is_contiguous(), \
            "bias must be contiguous"

        assert bias.dtype in (torch.float16, torch.bfloat16, torch.float32), \
            "bias dtype must be fp16, bf16, or fp32"

        bias_arg = bias
    else:
        # HAS_BIAS=False 时 kernel 不会访问 bias_ptr。
        # 这里传 c 只是占位，避免传 None。
        bias_arg = c

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _matmul_kernel[grid](
        a,
        b,
        bias_arg,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=HAS_BIAS,
    )

    return c