import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_mul_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if APPROXIMATE_TANH:
        gate_cubed = gate * gate * gate
        inner = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu = 0.5 * gate * (1.0 + tanh_inner)
    else:
        gelu = 0.5 * gate * (1.0 + tl.erf(gate * 0.7071067811865476))
    out = gelu * up

    tl.store(output_ptr + offsets, out, mask=mask)


def gelu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    output: torch.Tensor,
    approximate: str = "none",
):
    assert gate.is_cuda and up.is_cuda and output.is_cuda, "All tensors must be on CUDA"
    assert gate.shape == up.shape == output.shape, (
        "All tensors must have the same shape"
    )
    assert gate.is_contiguous() and up.is_contiguous() and output.is_contiguous(), (
        "All tensors must be contiguous"
    )
    assert gate.dtype in (torch.float16, torch.bfloat16), (
        "gate dtype must be fp16 or bf16"
    )
    assert up.dtype == gate.dtype and output.dtype == gate.dtype, (
        "gate, up, and output must have the same dtype"
    )
    assert approximate in ("none", "tanh"), 'approximate must be "none" or "tanh"'

    n_elements = gate.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _gelu_mul_kernel[grid](
        gate,
        up,
        output,
        n_elements,
        BLOCK_SIZE=1024,
        APPROXIMATE_TANH=approximate == "tanh",
    )
