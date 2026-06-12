import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x + y

    tl.store(output_ptr + offsets, out, mask=mask)

def vector_add(x, y, output):
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)