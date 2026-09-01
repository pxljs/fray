import math
import torch
import torch.nn.functional as F
import triton

import fray
from fray import bench_kineto


RTOL = 2e-2
ATOL = 2.0


G1_DEFAULT = {
    "name": "g1_BM64_BN128_BK32_GM8_W4_S2",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

G2_DEFAULT = {
    "name": "g2_BM64_BN256_BK32_GM8_W8_S3",
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}

PAIR = {"name": "large_h", "g1": G1_DEFAULT, "g2": G2_DEFAULT}


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------

def make_topk_assignments(num_tokens, num_experts, top_k, mode, device="cuda"):
    assert top_k <= num_experts
    tokens = torch.arange(num_tokens, device=device, dtype=torch.int64)

    if mode == "uniform":
        topk_ids = torch.empty((num_tokens, top_k), device=device, dtype=torch.int64)
        for k in range(top_k):
            topk_ids[:, k] = (tokens + k) % num_experts

    elif mode == "with_empty":
        active = max(top_k, num_experts // 2)
        topk_ids = torch.empty((num_tokens, top_k), device=device, dtype=torch.int64)
        for k in range(top_k):
            topk_ids[:, k] = (tokens + k) % active

    elif mode == "skewed":
        probs = torch.ones((num_experts,), device=device, dtype=torch.float32)
        if num_experts >= 1:
            probs[0] = 16.0
        if num_experts >= 2:
            probs[1] = 8.0
        if num_experts >= 3:
            probs[2] = 4.0
        if num_experts >= 4:
            probs[3] = 2.0

        topk_ids = torch.multinomial(
            probs,
            num_samples=num_tokens * top_k,
            replacement=True,
        ).view(num_tokens, top_k)

        # Test-data construction only; not benchmarked.
        for t in range(num_tokens):
            used = set()
            for k in range(top_k):
                e = int(topk_ids[t, k].item())
                while e in used:
                    e = (e + 1) % num_experts
                used.add(e)
                topk_ids[t, k] = e
    else:
        raise ValueError(f"Unknown routing mode: {mode}")

    raw = torch.rand((num_tokens, top_k), device=device, dtype=torch.float32)
    topk_weights = raw / raw.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.contiguous()


def init_moe_tensors(num_tokens, num_experts, H, I, dtype=torch.float16, device="cuda"):
    x = torch.randn((num_tokens, H), device=device, dtype=dtype) * 0.02
    w13 = torch.randn((num_experts, H, 2 * I), device=device, dtype=dtype) / math.sqrt(H)
    w2 = torch.randn((num_experts, I, H), device=device, dtype=dtype) / math.sqrt(I)
    return x.contiguous(), w13.contiguous(), w2.contiguous()


def make_case(num_tokens, num_experts, H, I, top_k, routing, dtype=torch.float16):
    topk_ids, topk_weights = make_topk_assignments(
        num_tokens, num_experts, top_k, routing, device="cuda"
    )
    x, w13, w2 = init_moe_tensors(
        num_tokens, num_experts, H, I, dtype=dtype, device="cuda"
    )

    sorted_token_ids, sorted_token_weights, expert_offsets, counts = (
        fray.triton.build_moe_dispatch_metadata_fast(
            topk_ids, topk_weights, num_experts
        )
    )

    return {
        "x": x,
        "w13": w13,
        "w2": w2,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "sorted_token_ids": sorted_token_ids,
        "sorted_token_weights": sorted_token_weights,
        "expert_offsets": expert_offsets,
        "counts": counts,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "hidden_size": H,
        "intermediate_size": I,
        "top_k": top_k,
        "routing": routing,
        "dtype": dtype,
    }


# ----------------------------------------------------------------------
# Reference / validation
# ----------------------------------------------------------------------

def torch_fused_moe_ref_indirect(case, output):
    x = case["x"]
    w13 = case["w13"]
    w2 = case["w2"]
    sorted_token_ids = case["sorted_token_ids"]
    sorted_token_weights = case["sorted_token_weights"]
    expert_offsets = case["expert_offsets"]
    E = case["num_experts"]
    I = case["intermediate_size"]

    output.zero_()

    for e in range(E):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        if end <= start:
            continue

        token_ids = sorted_token_ids[start:end]
        weights = sorted_token_weights[start:end]
        x_e = x[token_ids]

        tmp13 = torch.mm(x_e, w13[e])
        gate = tmp13[:, :I]
        up = tmp13[:, I:]
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        expert_out = torch.mm(hidden, w2[e])

        output.index_add_(
            0,
            token_ids,
            (expert_out.float() * weights.float()[:, None]).to(output.dtype),
        )
    return output


def compare_outputs(out, ref):
    torch.cuda.synchronize()
    out_f = out.float()
    ref_f = ref.float()
    finite = torch.isfinite(out_f) & torch.isfinite(ref_f)

    if finite.any():
        diff = torch.abs(out_f[finite] - ref_f[finite])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
    else:
        max_diff = float("nan")
        mean_diff = float("nan")

    ok = torch.isfinite(out_f).all().item() and torch.isfinite(ref_f).all().item()
    ok = bool(ok and torch.allclose(out, ref, rtol=RTOL, atol=ATOL))
    return {"ok": ok, "max_diff": max_diff, "mean_diff": mean_diff}


# ----------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------

def build_new_tile_metadata(case, pair=PAIR):
    expert_offsets = case["expert_offsets"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    g1 = pair["g1"]
    g2 = pair["g2"]

    g1_meta = fray.triton.build_grouped_tile_offsets(
        expert_offsets,
        I,
        BLOCK_M=g1["BLOCK_M"],
        BLOCK_N=g1["BLOCK_N"],
    )
    g2_meta = fray.triton.build_grouped_tile_offsets(
        expert_offsets,
        H,
        BLOCK_M=g2["BLOCK_M"],
        BLOCK_N=g2["BLOCK_N"],
    )
    return g1_meta, g2_meta


def build_new_tile_metadata_no_item_like(case, pair=PAIR):
    """
    Measures tensor-side work similar to build_grouped_tile_offsets but without the final .item().
    This helps estimate whether Python scalar sync is a large part of metadata time.
    """
    expert_offsets = case["expert_offsets"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    metas = []
    for N, cfg in [(I, pair["g1"]), (H, pair["g2"])]:
        counts = expert_offsets[1:] - expert_offsets[:-1]
        m_tiles = torch.div(
            counts + cfg["BLOCK_M"] - 1,
            cfg["BLOCK_M"],
            rounding_mode="floor",
        )
        n_tiles = triton.cdiv(N, cfg["BLOCK_N"])
        tiles_per_expert = m_tiles * n_tiles
        offsets = torch.empty_like(expert_offsets)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(tiles_per_expert, dim=0)
        metas.append(offsets)
    return tuple(metas)


def build_old_per_tile_metadata_if_available(case, pair=PAIR):
    if not hasattr(fray.triton, "build_grouped_gemm_metadata"):
        return None

    expert_offsets = case["expert_offsets"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    g1 = pair["g1"]
    g2 = pair["g2"]

    g1_meta = fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        I,
        BLOCK_M=g1["BLOCK_M"],
        BLOCK_N=g1["BLOCK_N"],
        GROUP_M=g1["GROUP_M"],
    )
    g2_meta = fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        H,
        BLOCK_M=g2["BLOCK_M"],
        BLOCK_N=g2["BLOCK_N"],
        GROUP_M=g2["GROUP_M"],
    )
    return g1_meta, g2_meta


def tensor_bytes(x):
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (tuple, list)):
        return sum(tensor_bytes(v) for v in x)
    return 0


# ----------------------------------------------------------------------
# Core benchmark blocks
# ----------------------------------------------------------------------

def run_prepared(case, metadata, output, hidden, pair=PAIR):
    use_atomic = case["top_k"] != 1
    clear_output = use_atomic
    g1_meta, g2_meta = metadata
    fray.triton.fused_moe_prepared(
        case["x"],
        case["w13"],
        case["w2"],
        case["sorted_token_ids"],
        case["sorted_token_weights"],
        case["expert_offsets"],
        output=output,
        hidden=hidden,
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        gemm1_tile_metadata=g1_meta,
        gemm2_tile_metadata=g2_meta,
        use_atomic=use_atomic,
        clear_output=clear_output,
    )


def metadata_builder_compare(case, pair=PAIR):
    print("\n" + "-" * 100)
    print("Metadata builder comparison")

    def run_new():
        return build_new_tile_metadata(case, pair)

    def run_new_no_item_like():
        return build_new_tile_metadata_no_item_like(case, pair)

    def run_old():
        return build_old_per_tile_metadata_if_available(case, pair)

    # warmup
    new_meta = run_new()
    new_no_item = run_new_no_item_like()
    old_meta = run_old()
    torch.cuda.synchronize()

    t_new = bench_kineto(run_new, "new_tile_offsets_with_item")
    t_new_no_item = bench_kineto(run_new_no_item_like, "new_tile_offsets_no_item_like")

    print(f"new tile offsets metadata    : {t_new * 1e6:8.2f} us")
    print(f"new no-item-like metadata    : {t_new_no_item * 1e6:8.2f} us")
    print(f"estimated item/sync overhead : {(t_new - t_new_no_item) * 1e6:8.2f} us")
    print(f"new metadata bytes           : {tensor_bytes(new_meta):8d} bytes")
    print(f"G1/G2 tiles                  : {new_meta[0][1]} / {new_meta[1][1]}")

    if old_meta is not None:
        t_old = bench_kineto(run_old, "old_per_tile_metadata")
        old_g1_tiles = old_meta[0][0].numel()
        old_g2_tiles = old_meta[1][0].numel()
        print(f"old per-tile metadata        : {t_old * 1e6:8.2f} us")
        print(f"old metadata bytes           : {tensor_bytes(old_meta):8d} bytes")
        print(f"old G1/G2 tiles              : {old_g1_tiles} / {old_g2_tiles}")
        print(f"new/old metadata time ratio  : {t_new / t_old:8.3f}x")
    else:
        print("old per-tile metadata        : unavailable; fray.triton.build_grouped_gemm_metadata not found")

    return new_meta


def kernel_breakdown(case, metadata, pair=PAIR):
    print("\n" + "-" * 100)
    print("Prepared kernel breakdown: GEMM1 indirect vs GEMM2 combine")

    num_tokens = case["num_tokens"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    T = num_tokens * case["top_k"]
    use_atomic = case["top_k"] != 1

    hidden = torch.empty((T, I), device="cuda", dtype=case["dtype"])
    output = torch.empty((num_tokens, H), device="cuda", dtype=case["dtype"])
    g1_meta, g2_meta = metadata

    def run_gemm1():
        fray.triton.moe_gemm1_silu_indirect(
            case["x"],
            case["w13"],
            case["sorted_token_ids"],
            hidden,
            case["expert_offsets"],
            tile_metadata=g1_meta,
            cfg=pair["g1"],
        )

    def run_gemm2():
        if use_atomic:
            output.zero_()
        fray.triton.moe_gemm2_combine(
            hidden,
            case["w2"],
            case["sorted_token_ids"],
            case["sorted_token_weights"],
            output,
            case["expert_offsets"],
            tile_metadata=g2_meta,
            cfg=pair["g2"],
            use_atomic=use_atomic,
        )

    # warmup: GEMM2 needs valid hidden.
    run_gemm1()
    run_gemm2()
    torch.cuda.synchronize()

    t_g1 = bench_kineto(run_gemm1, "gemm1_silu_indirect_only")
    # Refresh hidden before timing g2, then benchmark g2 only.
    run_gemm1()
    torch.cuda.synchronize()
    t_g2 = bench_kineto(run_gemm2, "gemm2_combine_only")

    print(f"GEMM1 indirect only          : {t_g1 * 1e6:8.2f} us")
    print(f"GEMM2 combine only           : {t_g2 * 1e6:8.2f} us")
    print(f"GEMM1 % of G1+G2             : {t_g1 / (t_g1 + t_g2) * 100:8.2f}%")
    print(f"GEMM2 % of G1+G2             : {t_g2 / (t_g1 + t_g2) * 100:8.2f}%")
    return t_g1, t_g2


def e2e_breakdown(case, metadata, pair=PAIR):
    print("\n" + "-" * 100)
    print("Full path breakdown")

    num_tokens = case["num_tokens"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    T = num_tokens * case["top_k"]
    use_atomic = case["top_k"] != 1
    clear_output = use_atomic

    output_prepared = torch.empty((num_tokens, H), device="cuda", dtype=case["dtype"])
    output_public = torch.empty_like(output_prepared)
    output_ref = torch.empty_like(output_prepared)
    hidden = torch.empty((T, I), device="cuda", dtype=case["dtype"])

    g1_meta, g2_meta = metadata

    def run_dispatch():
        return fray.triton.build_moe_dispatch_metadata_fast(
            case["topk_ids"], case["topk_weights"], case["num_experts"]
        )

    def run_tile_metadata():
        return build_new_tile_metadata(case, pair)

    def run_prepared_block():
        run_prepared(case, metadata, output_prepared, hidden, pair)

    def run_public():
        fray.triton.fused_moe(
            case["x"],
            case["topk_ids"],
            case["topk_weights"],
            case["w13"],
            case["w2"],
            output=output_public,
            hidden=hidden,
            gemm1_cfg=pair["g1"],
            gemm2_cfg=pair["g2"],
            use_atomic=use_atomic,
            clear_output=clear_output,
        )

    torch_fused_moe_ref_indirect(case, output_ref)

    # warmup / JIT
    run_dispatch()
    run_tile_metadata()
    run_prepared_block()
    run_public()
    torch.cuda.synchronize()

    stats_prepared = compare_outputs(output_prepared, output_ref)
    stats_public = compare_outputs(output_public, output_ref)

    t_dispatch = bench_kineto(run_dispatch, "dispatch_metadata_fast")
    t_tile = bench_kineto(run_tile_metadata, "grouped_tile_offsets")
    t_prepared = bench_kineto(run_prepared_block, "fused_moe_prepared_kernel")
    t_public = bench_kineto(run_public, "fused_moe_public_e2e")

    accounted = t_dispatch + t_tile + t_prepared
    unaccounted = t_public - accounted

    print(f"prepared accuracy            : {'PASS' if stats_prepared['ok'] else 'FAIL'}")
    print(f"public accuracy              : {'PASS' if stats_public['ok'] else 'FAIL'}")
    print(f"dispatch metadata            : {t_dispatch * 1e6:8.2f} us")
    print(f"tile metadata                : {t_tile * 1e6:8.2f} us")
    print(f"prepared kernel              : {t_prepared * 1e6:8.2f} us")
    print(f"public e2e                   : {t_public * 1e6:8.2f} us")
    print(f"accounted sum                : {accounted * 1e6:8.2f} us")
    print(f"unaccounted                  : {unaccounted * 1e6:8.2f} us")
    print(f"dispatch %                   : {t_dispatch / t_public * 100:8.2f}%")
    print(f"tile metadata %              : {t_tile / t_public * 100:8.2f}%")
    print(f"prepared kernel %            : {t_prepared / t_public * 100:8.2f}%")
    print(f"prepared max diff            : {stats_prepared['max_diff']:.6f}")
    print(f"public max diff              : {stats_public['max_diff']:.6f}")

    return {
        "dispatch_us": t_dispatch * 1e6,
        "tile_metadata_us": t_tile * 1e6,
        "prepared_us": t_prepared * 1e6,
        "public_us": t_public * 1e6,
    }


def print_case_summary(case, metadata):
    counts = case["counts"].detach().cpu().tolist()
    active = sum(1 for v in counts if v > 0)
    g1_meta, g2_meta = metadata
    print("\n" + "=" * 100)
    print(
        f"N={case['num_tokens']}, E={case['num_experts']}, "
        f"H={case['hidden_size']}, I={case['intermediate_size']}, "
        f"top_k={case['top_k']}, routing={case['routing']}"
    )
    print(f"tokens                       : {case['num_tokens']}")
    print(f"expanded                     : {case['num_tokens'] * case['top_k']}")
    print(f"active experts               : {active}")
    print(f"max/expert                   : {max(counts) if counts else 0}")
    print(f"counts sample                : {counts[:min(16, len(counts))]}")
    print(f"new G1/G2 tiles              : {g1_meta[1]} / {g2_meta[1]}")


# ----------------------------------------------------------------------
# Main diagnostic suite
# ----------------------------------------------------------------------

def run_diagnostics():
    torch.manual_seed(42)
    cases = [
        (512, 8, 1024, 2816, 1, "uniform"),
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "with_empty"),
    ]

    print("\n" + "=" * 100)
    print("Fused MoE Diagnostics")
    print("=" * 100)
    print("This script diagnoses:")
    print("  1. old per-tile metadata vs new expert_tile_offsets metadata build cost")
    print("  2. .item()/CPU-sync contribution in new metadata")
    print("  3. GEMM1 indirect vs GEMM2 combine kernel time")
    print("  4. prepared kernel vs public e2e")

    for spec in cases:
        case = make_case(*spec)
        metadata = build_new_tile_metadata(case, PAIR)
        print_case_summary(case, metadata)
        metadata_builder_compare(case, PAIR)
        kernel_breakdown(case, metadata, PAIR)
        e2e_breakdown(case, metadata, PAIR)


if __name__ == "__main__":
    run_diagnostics()
