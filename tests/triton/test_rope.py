import torch
import triton

import fray
from fray import bench_kineto


def precompute_rope_params(max_seq_len, head_dim, theta=10000.0, device="cuda"):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


def torch_rope_reference(
    q, k, cos, sin, positions, rotary_dim=None, is_neox_style=True
):
    def apply_rotary(x):
        if rotary_dim is None:
            actual_rotary_dim = x.shape[-1]
        else:
            actual_rotary_dim = rotary_dim
        half_dim = actual_rotary_dim // 2
        c = cos[positions].unsqueeze(1).to(torch.float32)
        s = sin[positions].unsqueeze(1).to(torch.float32)

        x_rotary = x[..., :actual_rotary_dim]
        x_tail = x[..., actual_rotary_dim:]
        if is_neox_style:
            x0 = x_rotary[..., :half_dim].to(torch.float32)
            x1 = x_rotary[..., half_dim:].to(torch.float32)
        else:
            x0 = x_rotary[..., 0::2].to(torch.float32)
            x1 = x_rotary[..., 1::2].to(torch.float32)

        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        if is_neox_style:
            y_rotary = torch.cat([y0, y1], dim=-1)
        else:
            y_rotary = torch.stack([y0, y1], dim=-1).flatten(-2)

        return torch.cat([y_rotary, x_tail.to(torch.float32)], dim=-1).to(x.dtype)

    return apply_rotary(q), apply_rotary(k)


def compare_outputs(q_out, k_out, q_ref, k_ref):
    q_max_diff = torch.max(torch.abs(q_out - q_ref)).item()
    k_max_diff = torch.max(torch.abs(k_out - k_ref)).item()
    q_mean_diff = torch.mean(torch.abs(q_out - q_ref)).item()
    k_mean_diff = torch.mean(torch.abs(k_out - k_ref)).item()
    is_close = torch.allclose(q_out, q_ref, rtol=1e-2, atol=1e-2) and torch.allclose(
        k_out, k_ref, rtol=1e-2, atol=1e-2
    )
    return {
        "ok": is_close,
        "q_max_diff": q_max_diff,
        "k_max_diff": k_max_diff,
        "q_mean_diff": q_mean_diff,
        "k_mean_diff": k_mean_diff,
    }


def rope_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton RoPE vs PyTorch RoPE")
    print("=" * 60)

    test_cases = [
        (1, 32, 32, 64, torch.float16),
        (128, 32, 32, 64, torch.float16),
        (1024, 32, 8, 128, torch.float16),
        (1024, 64, 8, 128, torch.float16),
        (256, 16, 4, 96, torch.bfloat16),
    ]

    for num_tokens, num_q_heads, num_kv_heads, head_dim, dtype in test_cases:
        q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
        k = torch.randn(
            (num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype
        )
        positions = torch.randint(
            0, 2048, (num_tokens,), device="cuda", dtype=torch.int32
        )
        cos, sin = precompute_rope_params(2048, head_dim)

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        fray.triton.rope(q, k, cos, sin, positions, q_out, k_out)
        q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
        torch.cuda.synchronize()

        stats = compare_outputs(q_out, k_out, q_ref, k_ref)

        print(
            f"\nShape: T={num_tokens}, Hq={num_q_heads}, "
            f"Hkv={num_kv_heads}, D={head_dim}, dtype={dtype}"
        )
        print(f"Q Triton Sample: {q_out[0, 0, :4].tolist()}")
        print(f"Q Torch Sample : {q_ref[0, 0, :4].tolist()}")
        print(f"Accuracy       : {'PASS' if stats['ok'] else 'FAIL'}")
        print(f"Q Max Diff     : {stats['q_max_diff']:.6f}")
        print(f"K Max Diff     : {stats['k_max_diff']:.6f}")

        assert stats["ok"]

    print("=" * 60 + "\n")


def test_rope_accuracy():
    rope_accuracy_test()


def test_rope_default_positions():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 256, 32, 8, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(num_tokens, head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    fray.triton.rope(q, k, cos, sin, None, q_out, k_out)
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_allocates_outputs():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 256, 16, 4, 96
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 1024, (num_tokens,), device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(1024, head_dim)

    q_out, k_out = fray.triton.rope(q, k, cos, sin, positions)
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_partial_rotary_dim():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 512, 32, 8
    head_dim, rotary_dim = 128, 64
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int64)
    cos, sin = precompute_rope_params(2048, rotary_dim)

    q_out, k_out = fray.triton.rope(q, k, cos, sin, positions, rotary_dim=rotary_dim)
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions, rotary_dim)
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_interleaved_style():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 512, 32, 8
    head_dim, rotary_dim = 128, 64
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(2048, rotary_dim)

    q_out, k_out = fray.triton.rope(
        q,
        k,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )
    q_ref, k_ref = torch_rope_reference(
        q,
        k,
        cos,
        sin,
        positions,
        rotary_dim,
        is_neox_style=False,
    )
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_full_interleaved_style():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 512, 32, 8, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(2048, head_dim)

    q_out, k_out = fray.triton.rope(
        q,
        k,
        cos,
        sin,
        positions,
        is_neox_style=False,
    )
    q_ref, k_ref = torch_rope_reference(
        q,
        k,
        cos,
        sin,
        positions,
        is_neox_style=False,
    )
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_grouped_full_head_tail_mask():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 257, 30, 7, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(2048, head_dim)

    q_out, k_out = fray.triton.rope(q, k, cos, sin, positions)
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_out, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_inplace():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 256, 32, 8, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    q_ref_src = q.clone()
    k_ref_src = k.clone()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cos, sin = precompute_rope_params(num_tokens, head_dim)

    q_out, k_out = fray.triton.rope_(q, k, cos, sin, positions)
    q_ref, k_ref = torch_rope_reference(q_ref_src, k_ref_src, cos, sin, positions)
    torch.cuda.synchronize()

    assert q_out is q
    assert k_out is k
    stats = compare_outputs(q, k, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_partial_inplace():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 256, 32, 8
    head_dim, rotary_dim = 128, 64
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    q_ref_src = q.clone()
    k_ref_src = k.clone()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(num_tokens, rotary_dim)

    q_out, k_out = fray.triton.rope_(q, k, cos, sin, positions, rotary_dim=rotary_dim)
    q_ref, k_ref = torch_rope_reference(
        q_ref_src,
        k_ref_src,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
    )
    torch.cuda.synchronize()

    assert q_out is q
    assert k_out is k
    stats = compare_outputs(q, k, q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_with_k_cache_full():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 256, 32, 8, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    cache_positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32) + 17
    cos, sin = precompute_rope_params(2048, head_dim)

    q_out = torch.empty_like(q)
    k_cache = torch.empty(
        (num_tokens + 32, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        cache_positions=cache_positions,
        q_out=q_out,
    )
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_cache[cache_positions], q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_with_k_cache_full_head_tail_mask():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads, head_dim = 257, 30, 7, 128
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    cache_positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32) + 5
    cos, sin = precompute_rope_params(2048, head_dim)

    k_cache = torch.empty(
        (num_tokens + 8, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        cache_positions=cache_positions,
    )
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_cache[cache_positions], q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_with_k_cache_partial_interleaved():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 257, 30, 7
    head_dim, rotary_dim = 128, 64
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int64)
    cache_positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64) + 3
    cos, sin = precompute_rope_params(2048, rotary_dim)

    k_cache = torch.empty(
        (num_tokens + 8, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        cache_positions=cache_positions,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )
    q_ref, k_ref = torch_rope_reference(
        q,
        k,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )
    torch.cuda.synchronize()

    stats = compare_outputs(q_out, k_cache[cache_positions], q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_with_k_cache_q_inplace():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 256, 32, 8
    head_dim, rotary_dim = 128, 64
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    q_ref_src = q.clone()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(num_tokens, rotary_dim)
    k_cache = torch.empty(
        (num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        q_out=q,
        rotary_dim=rotary_dim,
    )
    q_ref, k_ref = torch_rope_reference(
        q_ref_src,
        k,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
    )
    torch.cuda.synchronize()

    assert q_out is q
    stats = compare_outputs(q, k_cache[positions], q_ref, k_ref)
    assert stats["ok"], stats


def test_rope_with_paged_k_cache_partial_interleaved():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 257, 30, 7
    head_dim, rotary_dim, page_size = 128, 64, 16
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    slot_ids = torch.randperm(num_tokens, device="cuda", dtype=torch.int64)
    page_indices = slot_ids // page_size
    page_offsets = slot_ids % page_size
    num_pages = triton.cdiv(num_tokens, page_size)
    cos, sin = precompute_rope_params(2048, rotary_dim)

    k_cache = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_paged_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        page_indices,
        page_offsets,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )
    q_ref, k_ref = torch_rope_reference(
        q,
        k,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
        is_neox_style=False,
    )
    torch.cuda.synchronize()

    k_cache_ref = k_cache.new_empty(k_cache.shape)
    k_cache_ref[page_indices, page_offsets] = k_ref
    stats = compare_outputs(
        q_out,
        k_cache[page_indices, page_offsets],
        q_ref,
        k_cache_ref[page_indices, page_offsets],
    )
    assert stats["ok"], stats


def test_rope_with_paged_k_cache_full():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 257, 30, 7
    head_dim, page_size = 128, 16
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.randint(0, 2048, (num_tokens,), device="cuda", dtype=torch.int32)
    slot_ids = torch.randperm(num_tokens, device="cuda", dtype=torch.int64)
    page_indices = slot_ids // page_size
    page_offsets = slot_ids % page_size
    num_pages = triton.cdiv(num_tokens, page_size)
    cos, sin = precompute_rope_params(2048, head_dim)

    k_cache = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )

    q_out, k_cache = fray.triton.rope_with_paged_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        page_indices,
        page_offsets,
    )
    q_ref, k_ref = torch_rope_reference(q, k, cos, sin, positions)
    torch.cuda.synchronize()

    stats = compare_outputs(
        q_out,
        k_cache[page_indices, page_offsets],
        q_ref,
        k_ref,
    )
    assert stats["ok"], stats


def test_rope_with_paged_k_cache_q_inplace():
    torch.manual_seed(42)

    num_tokens, num_q_heads, num_kv_heads = 256, 32, 8
    head_dim, rotary_dim, page_size = 128, 64, 32
    dtype = torch.float16

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    q_ref_src = q.clone()
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    slot_ids = torch.arange(num_tokens, device="cuda", dtype=torch.int64).flip(0)
    page_indices = slot_ids // page_size
    page_offsets = slot_ids % page_size
    cos, sin = precompute_rope_params(num_tokens, rotary_dim)

    k_cache = torch.empty(
        (triton.cdiv(num_tokens, page_size), page_size, num_kv_heads, head_dim),
        device="cuda",
        dtype=dtype,
    )

    q_out, k_cache = fray.triton.rope_with_paged_k_cache(
        q,
        k,
        cos,
        sin,
        positions,
        k_cache,
        page_indices,
        page_offsets,
        q_out=q,
        rotary_dim=rotary_dim,
    )
    q_ref, k_ref = torch_rope_reference(
        q_ref_src,
        k,
        cos,
        sin,
        positions,
        rotary_dim=rotary_dim,
    )
    torch.cuda.synchronize()

    assert q_out is q
    stats = compare_outputs(
        q,
        k_cache[page_indices, page_offsets],
        q_ref,
        k_ref,
    )
    assert stats["ok"], stats


def benchmark_rope_case(
    label,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim=None,
    is_neox_style=True,
):
    dtype = torch.float16
    actual_rotary_dim = head_dim if rotary_dim is None else rotary_dim

    print(
        f"\nConfiguration: {label}, T={num_tokens}, Hq={num_q_heads}, "
        f"Hkv={num_kv_heads}, D={head_dim}, rotary_dim={actual_rotary_dim}, "
        f"is_neox_style={is_neox_style}, dtype={dtype}"
    )

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(num_tokens, actual_rotary_dim)

    q_triton = torch.empty_like(q)
    k_triton = torch.empty_like(k)
    q_torch = torch.empty_like(q)
    k_torch = torch.empty_like(k)

    def run_triton():
        fray.triton.rope(
            q,
            k,
            cos,
            sin,
            positions,
            q_triton,
            k_triton,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )

    def run_torch():
        q_ref, k_ref = torch_rope_reference(
            q,
            k,
            cos,
            sin,
            positions,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        q_torch.copy_(q_ref)
        k_torch.copy_(k_ref)

    run_triton()
    run_torch()
    torch.cuda.synchronize()

    stats = compare_outputs(q_triton, k_triton, q_torch, k_torch)
    if not stats["ok"]:
        print(
            "WARNING: correctness mismatch, "
            f"q_max_diff={stats['q_max_diff']:.6f}, "
            f"k_max_diff={stats['k_max_diff']:.6f}"
        )

    t_torch_s = bench_kineto(run_torch, f"torch_rope_{label}")
    t_triton_s = bench_kineto(run_triton, f"triton_rope_{label}")

    element_bytes = q.element_size()
    qk_elements = q.numel() + k.numel()
    table_elements = num_tokens * actual_rotary_dim
    total_bytes = qk_elements * element_bytes * 2 + table_elements * cos.element_size()
    triton_gbps = total_bytes / t_triton_s / 1e9
    torch_gbps = total_bytes / t_torch_s / 1e9

    print("-" * 60)
    print(
        f"Triton RoPE : {t_triton_s * 1e6:8.2f} us | "
        f"Effective Bandwidth: {triton_gbps:8.2f} GB/s"
    )
    print(
        f"PyTorch RoPE: {t_torch_s * 1e6:8.2f} us | "
        f"Effective Bandwidth: {torch_gbps:8.2f} GB/s"
    )
    print(f"Speedup     : {t_torch_s / t_triton_s:8.2f}x")
    print(f"Q Max Diff  : {stats['q_max_diff']:.6f}")
    print(f"K Max Diff  : {stats['k_max_diff']:.6f}")
    print("-" * 60)


def benchmark_rope_inplace_case(
    label,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim=None,
    is_neox_style=True,
):
    dtype = torch.float16
    actual_rotary_dim = head_dim if rotary_dim is None else rotary_dim

    print(
        f"\nConfiguration: {label}, T={num_tokens}, Hq={num_q_heads}, "
        f"Hkv={num_kv_heads}, D={head_dim}, rotary_dim={actual_rotary_dim}, "
        f"is_neox_style={is_neox_style}, dtype={dtype}"
    )

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cos, sin = precompute_rope_params(num_tokens, actual_rotary_dim)

    def run_triton():
        fray.triton.rope_(
            q,
            k,
            cos,
            sin,
            positions,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )

    run_triton()
    torch.cuda.synchronize()

    t_triton_s = bench_kineto(run_triton, f"triton_rope_{label}")

    element_bytes = q.element_size()
    rotary_elements = num_tokens * (num_q_heads + num_kv_heads) * actual_rotary_dim
    table_elements = num_tokens * actual_rotary_dim
    total_bytes = (
        rotary_elements * element_bytes * 2 + table_elements * cos.element_size()
    )
    triton_gbps = total_bytes / t_triton_s / 1e9

    print("-" * 60)
    print(
        f"Triton RoPE : {t_triton_s * 1e6:8.2f} us | "
        f"Effective Bandwidth: {triton_gbps:8.2f} GB/s"
    )
    print("-" * 60)


def benchmark_rope_k_cache_case(
    label,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim=None,
    is_neox_style=True,
):
    dtype = torch.float16
    actual_rotary_dim = head_dim if rotary_dim is None else rotary_dim

    print(
        f"\nConfiguration: {label}, T={num_tokens}, Hq={num_q_heads}, "
        f"Hkv={num_kv_heads}, D={head_dim}, rotary_dim={actual_rotary_dim}, "
        f"is_neox_style={is_neox_style}, dtype={dtype}"
    )

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    cache_positions = positions + 16
    cos, sin = precompute_rope_params(num_tokens, actual_rotary_dim)

    q_fused = torch.empty_like(q)
    q_two_step = torch.empty_like(q)
    k_tmp = torch.empty_like(k)
    k_cache_fused = torch.empty(
        (num_tokens + 32, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )
    k_cache_two_step = torch.empty_like(k_cache_fused)

    def run_fused():
        fray.triton.rope_with_k_cache(
            q,
            k,
            cos,
            sin,
            positions,
            k_cache_fused,
            cache_positions=cache_positions,
            q_out=q_fused,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )

    def run_two_step():
        fray.triton.rope(
            q,
            k,
            cos,
            sin,
            positions,
            q_two_step,
            k_tmp,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        k_cache_two_step[cache_positions] = k_tmp

    run_fused()
    run_two_step()
    torch.cuda.synchronize()

    stats = compare_outputs(
        q_fused,
        k_cache_fused[cache_positions],
        q_two_step,
        k_cache_two_step[cache_positions],
    )
    if not stats["ok"]:
        print(
            "WARNING: correctness mismatch, "
            f"q_max_diff={stats['q_max_diff']:.6f}, "
            f"k_max_diff={stats['k_max_diff']:.6f}"
        )

    t_fused_s = bench_kineto(run_fused, f"triton_rope_k_cache_{label}")
    t_two_step_s = bench_kineto(run_two_step, f"triton_rope_then_cache_{label}")

    print("-" * 60)
    print(f"RoPE + K cache fused   : {t_fused_s * 1e6:8.2f} us")
    print(f"RoPE then K cache copy : {t_two_step_s * 1e6:8.2f} us")
    print(f"Speedup               : {t_two_step_s / t_fused_s:8.2f}x")
    print(f"Q Max Diff            : {stats['q_max_diff']:.6f}")
    print(f"K Max Diff            : {stats['k_max_diff']:.6f}")
    print("-" * 60)


def benchmark_rope_paged_k_cache_case(
    label,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    page_size,
    rotary_dim=None,
    is_neox_style=True,
):
    dtype = torch.float16
    actual_rotary_dim = head_dim if rotary_dim is None else rotary_dim

    print(
        f"\nConfiguration: {label}, T={num_tokens}, Hq={num_q_heads}, "
        f"Hkv={num_kv_heads}, D={head_dim}, rotary_dim={actual_rotary_dim}, "
        f"page_size={page_size}, is_neox_style={is_neox_style}, dtype={dtype}"
    )

    q = torch.randn((num_tokens, num_q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((num_tokens, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
    slot_ids = torch.randperm(num_tokens, device="cuda", dtype=torch.int64)
    page_indices = slot_ids // page_size
    page_offsets = slot_ids % page_size
    num_pages = triton.cdiv(num_tokens, page_size)
    cos, sin = precompute_rope_params(num_tokens, actual_rotary_dim)

    q_fused = torch.empty_like(q)
    q_two_step = torch.empty_like(q)
    k_tmp = torch.empty_like(k)
    k_cache_fused = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim), device="cuda", dtype=dtype
    )
    k_cache_two_step = torch.empty_like(k_cache_fused)

    def run_fused():
        fray.triton.rope_with_paged_k_cache(
            q,
            k,
            cos,
            sin,
            positions,
            k_cache_fused,
            page_indices,
            page_offsets,
            q_out=q_fused,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )

    def run_two_step():
        fray.triton.rope(
            q,
            k,
            cos,
            sin,
            positions,
            q_two_step,
            k_tmp,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
        k_cache_two_step[page_indices, page_offsets] = k_tmp

    run_fused()
    run_two_step()
    torch.cuda.synchronize()

    stats = compare_outputs(
        q_fused,
        k_cache_fused[page_indices, page_offsets],
        q_two_step,
        k_cache_two_step[page_indices, page_offsets],
    )
    if not stats["ok"]:
        print(
            "WARNING: correctness mismatch, "
            f"q_max_diff={stats['q_max_diff']:.6f}, "
            f"k_max_diff={stats['k_max_diff']:.6f}"
        )

    t_fused_s = bench_kineto(run_fused, f"triton_rope_paged_k_cache_{label}")
    t_two_step_s = bench_kineto(run_two_step, f"triton_rope_then_paged_cache_{label}")

    print("-" * 60)
    print(f"RoPE + paged K cache fused   : {t_fused_s * 1e6:8.2f} us")
    print(f"RoPE then paged K cache copy : {t_two_step_s * 1e6:8.2f} us")
    print(f"Speedup                     : {t_two_step_s / t_fused_s:8.2f}x")
    print(f"Q Max Diff                  : {stats['q_max_diff']:.6f}")
    print(f"K Max Diff                  : {stats['k_max_diff']:.6f}")
    print("-" * 60)


def test_rope_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton RoPE vs PyTorch RoPE")
    print("=" * 60)

    test_cases = [
        ("full_neox", 1024, 32, 32, 64, None, True),
        ("full_neox", 2048, 32, 32, 128, None, True),
        ("full_neox", 4096, 32, 8, 128, None, True),
        ("full_neox", 8192, 32, 8, 128, None, True),
        ("full_neox", 2048, 64, 8, 128, None, True),
        ("full_interleaved", 4096, 32, 8, 128, None, False),
        ("partial_neox", 4096, 32, 8, 128, 64, True),
        ("partial_interleaved", 4096, 32, 8, 128, 64, False),
    ]

    for case in test_cases:
        benchmark_rope_case(*case)

    benchmark_rope_inplace_case(
        "partial_neox_inplace",
        4096,
        32,
        8,
        128,
        rotary_dim=64,
        is_neox_style=True,
    )
    benchmark_rope_k_cache_case(
        "full_neox",
        4096,
        32,
        8,
        128,
    )
    benchmark_rope_k_cache_case(
        "partial_interleaved",
        4096,
        32,
        8,
        128,
        rotary_dim=64,
        is_neox_style=False,
    )
    benchmark_rope_paged_k_cache_case(
        "full_neox",
        4096,
        32,
        8,
        128,
        page_size=16,
    )
    benchmark_rope_paged_k_cache_case(
        "partial_interleaved",
        4096,
        32,
        8,
        128,
        page_size=16,
        rotary_dim=64,
        is_neox_style=False,
    )


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    rope_accuracy_test()

    print("\nStep 2: Running In-place Test...")
    test_rope_inplace()

    print("\nStep 3: Running Performance Test...")
    test_rope_performance()
