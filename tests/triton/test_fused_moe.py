import math

import torch
import torch.nn.functional as F

import fray
from fray import bench_kineto


RTOL = 2e-2
ATOL = 2.0


# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------

G1_LARGE_H = {
    "name": "g1_BM64_BN128_BK32_GM8_W4_S2",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

G2_LARGE_H = {
    "name": "g2_BM64_BN256_BK32_GM8_W8_S3",
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}

G1_SMALL_M = {
    "name": "g1_BM32_BN64_BK32_GM8_W4_S2",
    "BLOCK_M": 32,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

G2_SMALL_M = {
    "name": "g2_BM32_BN128_BK32_GM8_W4_S3",
    "BLOCK_M": 32,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}

G1_SAFE = {
    "name": "g1_BM32_BN32_BK32_GM8_W4_S1",
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 1,
}

DEFAULT_PAIR = {
    "name": "large_h",
    "g1": G1_LARGE_H,
    "g2": G2_LARGE_H,
}

CANDIDATE_PAIRS = [
    DEFAULT_PAIR,
    {"name": "small_m", "g1": G1_SMALL_M, "g2": G2_SMALL_M},
    {"name": "safe", "g1": G1_SAFE, "g2": G2_SMALL_M},
]


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------

def make_topk_assignments(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    mode: str,
    device: str = "cuda",
):
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

    elif mode == "single_hot":
        topk_ids = torch.empty((num_tokens, top_k), device=device, dtype=torch.int64)
        topk_ids[:, 0] = 0
        for k in range(1, top_k):
            topk_ids[:, k] = 1 + ((tokens + k - 1) % max(1, num_experts - 1))

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

        # Only data construction, not benchmarked.
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


def init_moe_tensors(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype=torch.float16,
    device="cuda",
):
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype) * 0.02
    w13 = torch.randn((num_experts, hidden_size, 2 * intermediate_size), device=device, dtype=dtype) / math.sqrt(hidden_size)
    w2 = torch.randn((num_experts, intermediate_size, hidden_size), device=device, dtype=dtype) / math.sqrt(intermediate_size)
    return x.contiguous(), w13.contiguous(), w2.contiguous()


def make_case(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    routing: str,
    dtype=torch.float16,
    device="cuda",
):
    topk_ids, topk_weights = make_topk_assignments(num_tokens, num_experts, top_k, routing, device=device)
    x, w13, w2 = init_moe_tensors(num_tokens, num_experts, hidden_size, intermediate_size, dtype=dtype, device=device)
    return {
        "x": x,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w13": w13,
        "w2": w2,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
        "routing": routing,
        "dtype": dtype,
    }


def alloc_workspaces(case):
    num_tokens = case["num_tokens"]
    top_k = case["top_k"]
    H = case["hidden_size"]
    I = case["intermediate_size"]
    E = case["num_experts"]
    T = num_tokens * top_k
    device = case["x"].device
    dtype = case["dtype"]

    return {
        "output": torch.empty((num_tokens, H), device=device, dtype=dtype),
        "ref": torch.empty((num_tokens, H), device=device, dtype=dtype),
        "hidden": torch.empty((T, I), device=device, dtype=dtype),
        "sorted_token_ids": torch.empty((T,), device=device, dtype=torch.int64),
        "sorted_token_weights": torch.empty((T,), device=device, dtype=case["topk_weights"].dtype),
        "expert_offsets": torch.empty((E + 1,), device=device, dtype=torch.int64),
        "expert_cursor": torch.empty((E,), device=device, dtype=torch.int64),
        "expert_counts": torch.empty((E,), device=device, dtype=torch.int64),
    }


# ----------------------------------------------------------------------
# Metadata / reference / validation
# ----------------------------------------------------------------------

def build_dispatch_metadata(case, workspace=None):
    kwargs = {}
    if workspace is not None:
        kwargs = {
            "sorted_token_ids": workspace["sorted_token_ids"],
            "sorted_token_weights": workspace["sorted_token_weights"],
            "expert_offsets": workspace["expert_offsets"],
            "expert_cursor": workspace["expert_cursor"],
        }

    sorted_token_ids, sorted_token_weights, expert_offsets, counts = (
        fray.triton.build_moe_dispatch_metadata_fast(
            case["topk_ids"],
            case["topk_weights"],
            case["num_experts"],
            **kwargs,
        )
    )

    return {
        "sorted_token_ids": sorted_token_ids,
        "sorted_token_weights": sorted_token_weights,
        "expert_offsets": expert_offsets,
        "counts": counts,
    }


def build_tile_metadata(case, dispatch, pair):
    expert_offsets = dispatch["expert_offsets"]
    I = case["intermediate_size"]
    H = case["hidden_size"]
    g1 = pair["g1"]
    g2 = pair["g2"]

    gemm1_meta = fray.triton.build_grouped_tile_offsets(
        expert_offsets,
        I,
        BLOCK_M=g1["BLOCK_M"],
        BLOCK_N=g1["BLOCK_N"],
    )

    gemm2_meta = fray.triton.build_grouped_tile_offsets(
        expert_offsets,
        H,
        BLOCK_M=g2["BLOCK_M"],
        BLOCK_N=g2["BLOCK_N"],
    )

    return gemm1_meta, gemm2_meta


def torch_fused_moe_ref(case, dispatch, output):
    x = case["x"]
    w13 = case["w13"]
    w2 = case["w2"]
    sorted_token_ids = dispatch["sorted_token_ids"]
    sorted_token_weights = dispatch["sorted_token_weights"]
    expert_offsets = dispatch["expert_offsets"]
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


def torch_fused_moe_full_ref_from_logits(case, router_logits, output):
    x = case["x"]
    w13 = case["w13"]
    w2 = case["w2"]
    E = case["num_experts"]
    I = case["intermediate_size"]
    top_k = case["top_k"]

    topk_vals, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)
    output.zero_()

    for e in range(E):
        token_ids, route_ids = torch.where(topk_ids == e)
        if token_ids.numel() == 0:
            continue

        weights = topk_weights[token_ids, route_ids]
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

    nonfinite_out = out.numel() - torch.isfinite(out_f).sum().item()
    nonfinite_ref = ref.numel() - torch.isfinite(ref_f).sum().item()

    if finite.any():
        diff = torch.abs(out_f[finite] - ref_f[finite])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
    else:
        max_diff = float("nan")
        mean_diff = float("nan")

    ok = (
        nonfinite_out == 0
        and nonfinite_ref == 0
        and torch.allclose(out, ref, rtol=RTOL, atol=ATOL)
    )

    return {
        "ok": ok,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "nonfinite_out": nonfinite_out,
        "nonfinite_ref": nonfinite_ref,
    }


# ----------------------------------------------------------------------
# Run helpers
# ----------------------------------------------------------------------

def run_prepared(case, dispatch, tile_meta, pair, output, hidden):
    use_atomic = case["top_k"] != 1
    clear_output = use_atomic
    gemm1_meta, gemm2_meta = tile_meta

    return fray.triton.fused_moe_prepared(
        case["x"],
        case["w13"],
        case["w2"],
        dispatch["sorted_token_ids"],
        dispatch["sorted_token_weights"],
        dispatch["expert_offsets"],
        output=output,
        hidden=hidden,
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        gemm1_tile_metadata=gemm1_meta,
        gemm2_tile_metadata=gemm2_meta,
        use_atomic=use_atomic,
        clear_output=clear_output,
    )


def run_public(case, pair, workspace, persistent_waves="auto"):
    use_atomic = case["top_k"] != 1
    clear_output = use_atomic

    return fray.triton.fused_moe(
        case["x"],
        case["topk_ids"],
        case["topk_weights"],
        case["w13"],
        case["w2"],
        output=workspace["output"],
        hidden=workspace["hidden"],
        sorted_token_ids=workspace["sorted_token_ids"],
        sorted_token_weights=workspace["sorted_token_weights"],
        expert_offsets=workspace["expert_offsets"],
        expert_cursor=workspace["expert_cursor"],
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        use_atomic=use_atomic,
        clear_output=clear_output,
        persistent_waves=persistent_waves,
    )


def run_public_logits(case, router_logits, pair, workspace, persistent_waves="auto"):
    use_atomic = case["top_k"] != 1
    clear_output = use_atomic

    return fray.triton.fused_moe(
        case["x"],
        None,
        None,
        case["w13"],
        case["w2"],
        router_logits=router_logits,
        top_k=case["top_k"],
        output=workspace["output"],
        hidden=workspace["hidden"],
        sorted_token_ids=workspace["sorted_token_ids"],
        sorted_token_weights=workspace["sorted_token_weights"],
        expert_offsets=workspace["expert_offsets"],
        expert_cursor=workspace["expert_cursor"],
        expert_counts=workspace["expert_counts"],
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        use_atomic=use_atomic,
        clear_output=clear_output,
        persistent_waves=persistent_waves,
    )


def warmup_public_autotune(case, pair, workspace):
    run_public(case, pair, workspace, persistent_waves="auto")
    run_public(case, pair, workspace, persistent_waves="auto")
    torch.cuda.synchronize()


def warmup_public_logits_autotune(case, router_logits, pair, workspace):
    run_public_logits(case, router_logits, pair, workspace, persistent_waves="auto")
    run_public_logits(case, router_logits, pair, workspace, persistent_waves="auto")
    torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def case_title(case):
    return (
        f"N={case['num_tokens']}, E={case['num_experts']}, "
        f"H={case['hidden_size']}, I={case['intermediate_size']}, "
        f"top_k={case['top_k']}, routing={case['routing']}"
    )


def print_distribution(case, dispatch):
    counts = dispatch["counts"].detach().cpu().tolist()
    active = sum(1 for x in counts if x > 0)
    print(f"tokens       : {case['num_tokens']}")
    print(f"top_k        : {case['top_k']}")
    print(f"expanded     : {case['num_tokens'] * case['top_k']}")
    print(f"experts      : {case['num_experts']}")
    print(f"active       : {active}")
    print(f"max/expert   : {max(counts) if counts else 0}")
    print(f"counts sample: {counts[:min(16, len(counts))]}")


def tile_counts(tile_meta):
    return tile_meta[0][1], tile_meta[1][1]


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def accuracy_test():
    torch.manual_seed(42)
    print("\n" + "=" * 80)
    print("Accuracy Test: indirect fused_moe")
    print("=" * 80)

    cases = [
        (32, 4, 128, 256, 1, "uniform"),
        (64, 4, 128, 256, 2, "uniform"),
        (127, 8, 255, 513, 2, "with_empty"),
        (256, 16, 1024, 2816, 2, "skewed"),
    ]

    for args in cases:
        case = make_case(*args)
        workspace = alloc_workspaces(case)
        pair = DEFAULT_PAIR

        dispatch = build_dispatch_metadata(case, workspace)
        tile_meta = build_tile_metadata(case, dispatch, pair)

        run_prepared(case, dispatch, tile_meta, pair, workspace["output"], workspace["hidden"])
        torch_fused_moe_ref(case, dispatch, workspace["ref"])

        stats = compare_outputs(workspace["output"], workspace["ref"])
        g1_tiles, g2_tiles = tile_counts(tile_meta)

        print("\n" + case_title(case))
        print_distribution(case, dispatch)
        print(f"config       : {pair['name']}")
        print(f"G1/G2 tiles  : {g1_tiles} / {g2_tiles}")
        print(f"accuracy     : {'PASS' if stats['ok'] else 'FAIL'}")
        print(f"max diff     : {stats['max_diff']:.6f}")
        print(f"mean diff    : {stats['mean_diff']:.6f}")
        print(f"nonfinite out: {stats['nonfinite_out']}")
        print(f"nonfinite ref: {stats['nonfinite_ref']}")


def routing_selection_test():
    torch.manual_seed(42)
    print("\n" + "=" * 80)
    print("Routing Selection Test: router_logits path")
    print("=" * 80)

    num_tokens, num_experts, top_k = 96, 8, 2
    logits = torch.randn((num_tokens, num_experts), device="cuda", dtype=torch.float32)

    topk_ids = torch.empty((num_tokens, top_k), device="cuda", dtype=torch.int64)
    topk_weights = torch.empty((num_tokens, top_k), device="cuda", dtype=torch.float32)
    counts = torch.empty((num_experts,), device="cuda", dtype=torch.int64)
    fray.triton.moe_select_topk_softmax_with_counts(
        logits,
        top_k,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        counts=counts,
    )

    ref_vals, ref_ids = torch.topk(logits, k=top_k, dim=-1)
    ref_weights = torch.softmax(ref_vals, dim=-1)
    ref_counts = torch.bincount(ref_ids.reshape(-1), minlength=num_experts)
    counted = fray.triton.moe_count_experts(topk_ids, num_experts)
    torch.cuda.synchronize()

    ids_ok = torch.equal(topk_ids, ref_ids)
    weights_ok = torch.allclose(topk_weights, ref_weights, rtol=1e-4, atol=1e-4)
    counts_ok = torch.equal(counts, ref_counts)
    count_kernel_ok = torch.equal(counted, ref_counts)
    print(f"topk ids     : {'PASS' if ids_ok else 'FAIL'}")
    print(f"topk weights : {'PASS' if weights_ok else 'FAIL'}")
    print(f"counts       : {'PASS' if counts_ok else 'FAIL'}")
    print(f"count kernel : {'PASS' if count_kernel_ok else 'FAIL'}")

    if not ids_ok or not weights_ok or not counts_ok or not count_kernel_ok:
        raise AssertionError("router top-k selection mismatch")

    case = make_case(num_tokens, num_experts, 128, 256, top_k, "uniform")
    case["topk_ids"] = topk_ids
    case["topk_weights"] = topk_weights
    workspace = alloc_workspaces(case)
    pair = DEFAULT_PAIR

    output = fray.triton.fused_moe(
        case["x"],
        None,
        None,
        case["w13"],
        case["w2"],
        router_logits=logits,
        top_k=top_k,
        output=workspace["output"],
        hidden=workspace["hidden"],
        sorted_token_ids=workspace["sorted_token_ids"],
        sorted_token_weights=workspace["sorted_token_weights"],
        expert_offsets=workspace["expert_offsets"],
        expert_cursor=workspace["expert_cursor"],
        expert_counts=counts,
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        persistent_waves=3,
    )

    dispatch = build_dispatch_metadata(case, workspace)
    torch_fused_moe_ref(case, dispatch, workspace["ref"])
    stats = compare_outputs(output, workspace["ref"])
    print(f"fused output : {'PASS' if stats['ok'] else 'FAIL'}")
    print(f"max diff     : {stats['max_diff']:.6f}")

    if not stats["ok"]:
        raise AssertionError("router_logits fused_moe output mismatch")


def perf_test():
    torch.manual_seed(42)
    print("\n" + "=" * 80)
    print("Performance Test: Triton fused MoE vs pure PyTorch MoE")
    print("=" * 80)
    print("Triton cached_auto excludes the first-run persistent_waves autotune cost.")

    cases = [
        (512, 8, 1024, 2816, 1, "uniform"),
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "skewed"),
    ]

    for args in cases:
        case = make_case(*args)
        workspace = alloc_workspaces(case)
        pair = DEFAULT_PAIR

        dispatch = build_dispatch_metadata(case, workspace)
        tile_meta = build_tile_metadata(case, dispatch, pair)

        def run_triton_prepared():
            run_prepared(case, dispatch, tile_meta, pair, workspace["output"], workspace["hidden"])

        def run_triton_topk_e2e():
            run_public(case, pair, workspace)

        def run_torch_prepared():
            torch_fused_moe_ref(case, dispatch, workspace["ref"])

        run_triton_prepared()
        warmup_public_autotune(case, pair, workspace)
        run_torch_prepared()
        torch.cuda.synchronize()

        stats = compare_outputs(workspace["output"], workspace["ref"])

        t_prepared = bench_kineto(run_triton_prepared, "fused_moe_prepared")
        t_torch_prepared = bench_kineto(run_torch_prepared, "torch_moe_prepared")
        t_topk_e2e = bench_kineto(run_triton_topk_e2e, "fused_moe_topk_e2e_cached_auto")

        router_logits = torch.randn(
            (case["num_tokens"], case["num_experts"]),
            device="cuda",
            dtype=torch.float32,
        )
        logits_case = dict(case)
        logits_vals, logits_ids = torch.topk(router_logits, k=case["top_k"], dim=-1)
        logits_case["topk_ids"] = logits_ids.contiguous()
        logits_case["topk_weights"] = torch.softmax(logits_vals, dim=-1).contiguous()
        logits_workspace = alloc_workspaces(logits_case)

        def run_triton_logits_e2e():
            run_public_logits(logits_case, router_logits, pair, logits_workspace)

        def run_torch_logits_e2e():
            torch_fused_moe_full_ref_from_logits(
                logits_case,
                router_logits,
                logits_workspace["ref"],
            )

        warmup_public_logits_autotune(logits_case, router_logits, pair, logits_workspace)
        run_torch_logits_e2e()
        torch.cuda.synchronize()
        logits_stats = compare_outputs(logits_workspace["output"], logits_workspace["ref"])

        t_logits_e2e = bench_kineto(run_triton_logits_e2e, "fused_moe_logits_e2e_cached_auto")
        t_torch_logits_e2e = bench_kineto(run_torch_logits_e2e, "torch_moe_logits_e2e")

        T = case["num_tokens"] * case["top_k"]
        H = case["hidden_size"]
        I = case["intermediate_size"]
        flops = 6.0 * T * H * I

        print("\n" + case_title(case))
        print_distribution(case, dispatch)
        print(f"config       : {pair['name']}")
        print(f"accuracy     : {'PASS' if stats['ok'] else 'FAIL'}")
        print(f"full accuracy: {'PASS' if logits_stats['ok'] else 'FAIL'}")
        print(f"triton full  : {t_logits_e2e * 1e6:8.2f} us | cached auto | {flops / t_logits_e2e / 1e12:8.2f} TFLOPS")
        print(f"torch full   : {t_torch_logits_e2e * 1e6:8.2f} us | pure PyTorch logits input")
        print(f"full speedup : {t_torch_logits_e2e / t_logits_e2e:8.2f}x")
        print(f"triton topk  : {t_topk_e2e * 1e6:8.2f} us | topk ids/weights input")
        print(f"compute only : triton={t_prepared * 1e6:8.2f} us, torch={t_torch_prepared * 1e6:8.2f} us, speedup={t_torch_prepared / t_prepared:6.2f}x")
        print(f"max diff     : {logits_stats['max_diff']:.6f}")
        print(f"mean diff    : {logits_stats['mean_diff']:.6f}")


def breakdown_test():
    torch.manual_seed(42)
    print("\n" + "=" * 80)
    print("Breakdown Test: dispatch / tile metadata / prepared kernel / public")
    print("=" * 80)
    print("public cached_auto excludes the first-run persistent_waves autotune cost.")

    cases = [
        (512, 8, 1024, 2816, 1, "uniform"),
        (512, 8, 1024, 2816, 2, "uniform"),
        (1024, 16, 2048, 5632, 2, "skewed"),
    ]

    for args in cases:
        case = make_case(*args)
        workspace = alloc_workspaces(case)
        pair = DEFAULT_PAIR

        def run_dispatch():
            return build_dispatch_metadata(case, workspace)

        dispatch = run_dispatch()

        def run_tile_metadata():
            return build_tile_metadata(case, dispatch, pair)

        tile_meta = run_tile_metadata()

        def run_tile_metadata_no_sync():
            expert_offsets = dispatch["expert_offsets"]
            total_tokens = case["num_tokens"] * case["top_k"]
            g1 = pair["g1"]
            g2 = pair["g2"]
            return fray.triton.build_grouped_tile_offsets_pair_no_sync(
                expert_offsets,
                case["intermediate_size"],
                case["hidden_size"],
                BLOCK_M1=g1["BLOCK_M"],
                BLOCK_N1=g1["BLOCK_N"],
                BLOCK_M2=g2["BLOCK_M"],
                BLOCK_N2=g2["BLOCK_N"],
                total_tokens=total_tokens,
            )

        tile_meta_no_sync = run_tile_metadata_no_sync()

        def run_prepared_kernel():
            run_prepared(case, dispatch, tile_meta, pair, workspace["output"], workspace["hidden"])

        def run_prepared_kernel_no_sync():
            run_prepared(case, dispatch, tile_meta_no_sync, pair, workspace["output"], workspace["hidden"])

        def run_public_fused_moe():
            run_public(case, pair, workspace)

        def run_public_fused_moe_w3():
            run_public(case, pair, workspace, persistent_waves=3)

        def run_public_fused_moe_pair():
            run_public(case, pair, workspace, persistent_waves=(3, 2))

        run_dispatch()
        run_tile_metadata()
        run_tile_metadata_no_sync()
        run_prepared_kernel()
        run_prepared_kernel_no_sync()
        warmup_public_autotune(case, pair, workspace)
        run_public_fused_moe_w3()
        run_public_fused_moe_pair()
        torch.cuda.synchronize()

        t_dispatch = bench_kineto(run_dispatch, "dispatch_metadata_fast")
        t_tile_meta = bench_kineto(run_tile_metadata, "grouped_tile_offsets_exact")
        t_tile_meta_no_sync = bench_kineto(run_tile_metadata_no_sync, "grouped_tile_offsets_no_sync")
        t_prepared = bench_kineto(run_prepared_kernel, "fused_moe_prepared_kernel")
        t_prepared_no_sync = bench_kineto(run_prepared_kernel_no_sync, "fused_moe_prepared_no_sync_grid")
        t_public = bench_kineto(run_public_fused_moe, "fused_moe_public_e2e_cached_auto")
        t_public_w3 = bench_kineto(run_public_fused_moe_w3, "fused_moe_public_e2e_w3")
        t_public_pair = bench_kineto(run_public_fused_moe_pair, "fused_moe_public_e2e_w3_w2")

        accounted_w3 = t_dispatch + t_tile_meta_no_sync + t_prepared_no_sync
        unaccounted_w3 = t_public_w3 - accounted_w3
        unaccounted_auto = t_public - accounted_w3
        g1_tiles, g2_tiles = tile_counts(tile_meta)
        g1_launch_programs, g2_launch_programs = tile_counts(tile_meta_no_sync)

        print("\n" + case_title(case))
        print_distribution(case, dispatch)
        print(f"G1/G2 tiles       : {g1_tiles} / {g2_tiles}")
        print(f"G1/G2 launch progs: {g1_launch_programs} / {g2_launch_programs}")
        print(f"dispatch metadata : {t_dispatch * 1e6:8.2f} us")
        print(f"tile exact        : {t_tile_meta * 1e6:8.2f} us")
        print(f"tile no-sync      : {t_tile_meta_no_sync * 1e6:8.2f} us")
        print(f"prepared exact    : {t_prepared * 1e6:8.2f} us")
        print(f"prepared no-sync  : {t_prepared_no_sync * 1e6:8.2f} us")
        print(f"public e2e        : {t_public * 1e6:8.2f} us")
        print(f"public e2e w3     : {t_public_w3 * 1e6:8.2f} us")
        print(f"public e2e w3/w2  : {t_public_pair * 1e6:8.2f} us")
        print(f"accounted w3 sum  : {accounted_w3 * 1e6:8.2f} us")
        print(f"unaccounted auto  : {unaccounted_auto * 1e6:8.2f} us")
        print(f"unaccounted w3    : {unaccounted_w3 * 1e6:8.2f} us")
        print(f"dispatch %        : {t_dispatch / t_public * 100:8.2f}%")
        print(f"tile no-sync %    : {t_tile_meta_no_sync / t_public * 100:8.2f}%")
        print(f"prepared no-sync %: {t_prepared_no_sync / t_public * 100:8.2f}%")


def robust_config_sweep():
    torch.manual_seed(42)
    print("\n" + "=" * 80)
    print("Robust Config Sweep: indirect fused_moe_prepared")
    print("=" * 80)

    cases = [
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "uniform"),
        (1024, 16, 2048, 5632, 2, "skewed"),
    ]

    summary = {p["name"]: {"times": [], "ratios": [], "wins": 0, "fail": 0} for p in CANDIDATE_PAIRS}

    for args in cases:
        case = make_case(*args)
        workspace = alloc_workspaces(case)

        dispatch = build_dispatch_metadata(case, workspace)
        torch_fused_moe_ref(case, dispatch, workspace["ref"])
        torch.cuda.synchronize()

        results = []
        print("\n" + "-" * 80)
        print(case_title(case))
        print_distribution(case, dispatch)

        for pair in CANDIDATE_PAIRS:
            try:
                tile_meta = build_tile_metadata(case, dispatch, pair)

                def run_candidate():
                    run_prepared(case, dispatch, tile_meta, pair, workspace["output"], workspace["hidden"])

                run_candidate()
                torch.cuda.synchronize()
                stats = compare_outputs(workspace["output"], workspace["ref"])
                if not stats["ok"]:
                    raise RuntimeError(f"accuracy failed, max_diff={stats['max_diff']:.6f}")

                t = bench_kineto(run_candidate, f"fused_moe_{pair['name']}")
                results.append((pair["name"], t, stats, tile_meta))

            except Exception as e:
                summary[pair["name"]]["fail"] += 1
                print(f"{pair['name']:>10}: FAIL | {type(e).__name__}: {str(e).splitlines()[0][:120]}")

        if not results:
            continue

        best_t = min(r[1] for r in results)
        best_name = min(results, key=lambda x: x[1])[0]

        for name, t, stats, tile_meta in results:
            ratio = t / best_t
            summary[name]["times"].append(t)
            summary[name]["ratios"].append(ratio)
            if name == best_name:
                summary[name]["wins"] += 1
            g1_tiles, g2_tiles = tile_counts(tile_meta)
            print(
                f"{name:>10}: {t * 1e6:8.2f} us | "
                f"ratio={ratio:5.3f} | "
                f"G1/G2 tiles={g1_tiles}/{g2_tiles} | "
                f"max_diff={stats['max_diff']:.6f}"
            )

    print("\n" + "=" * 80)
    print("Robust Config Summary")
    print("=" * 80)
    print(f"{'config':>10} | {'avg_ratio':>9} | {'worst':>9} | {'wins':>4} | {'fail':>4} | {'avg_us':>10}")
    print("-" * 64)

    rows = []
    for pair in CANDIDATE_PAIRS:
        name = pair["name"]
        item = summary[name]
        ratios = item["ratios"]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            worst = max(ratios)
            avg_us = sum(item["times"]) / len(item["times"]) * 1e6
        else:
            avg_ratio = float("inf")
            worst = float("inf")
            avg_us = float("inf")
        rows.append((avg_ratio, worst, -item["wins"], item["fail"], avg_us, name))

    rows.sort()
    for avg_ratio, worst, neg_wins, fail, avg_us, name in rows:
        print(f"{name:>10} | {avg_ratio:9.3f} | {worst:9.3f} | {-neg_wins:4d} | {fail:4d} | {avg_us:10.2f}")


if __name__ == "__main__":
    # accuracy_test()
    # routing_selection_test()
    perf_test()
    # breakdown_test()
    # robust_config_sweep()
