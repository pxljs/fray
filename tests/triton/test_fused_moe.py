#!/usr/bin/env python3

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

import fray
from fray import bench_kineto


RTOL = 2e-2
ATOL = 2.0


# ----------------------------------------------------------------------
# Candidate configs
# ----------------------------------------------------------------------

G1_BALANCED = {
    "name": "g1_BM64_BN128_BK32_GM8_W4_S2",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
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

G1_SAFE = {
    "name": "g1_BM32_BN32_BK32_GM8_W4_S1",
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 1,
}

G2_BALANCED = {
    "name": "g2_BM128_BN128_BK32_GM8_W4_S3",
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 3,
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

G2_LARGE_H = {
    "name": "g2_BM64_BN256_BK32_GM8_W8_S3",
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}

DEFAULT_GEMM1_SILU_CFG = {
    "name": "g1_BM64_BN128_BK32_GM8_W4_S2",
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}

DEFAULT_GEMM2_COMBINE_CFG = {
    "name": "g2_BM64_BN256_BK32_GM8_W8_S3",
    "BLOCK_M": 64,
    "BLOCK_N": 256,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}

# 默认主配置：你当前 sweep 出来的 best。
DEFAULT_PAIR = {
    "name": "large_h",
    "g1": DEFAULT_GEMM1_SILU_CFG,
    "g2": DEFAULT_GEMM2_COMBINE_CFG,
}


CANDIDATE_PAIRS = [
    DEFAULT_PAIR,
    {
        "name": "small_m",
        "g1": G1_SMALL_M,
        "g2": G2_SMALL_M,
    },
    {
        "name": "large_h",
        "g1": G1_BALANCED,
        "g2": G2_LARGE_H,
    },
    {
        "name": "safe",
        "g1": G1_SAFE,
        "g2": G2_SMALL_M,
    },
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

        # 测试数据构造阶段，不计入 benchmark。
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
    x = torch.randn(
        (num_tokens, hidden_size),
        device=device,
        dtype=dtype,
    ) * 0.02

    w13 = torch.randn(
        (num_experts, hidden_size, 2 * intermediate_size),
        device=device,
        dtype=dtype,
    ) / math.sqrt(hidden_size)

    w2 = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    ) / math.sqrt(intermediate_size)

    return x.contiguous(), w13.contiguous(), w2.contiguous()


def build_sorted_moe_inputs(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
):
    num_tokens, _ = x.shape
    _, top_k = topk_ids.shape

    expanded_token_ids = torch.arange(
        num_tokens,
        device=x.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)

    expanded_expert_ids = topk_ids.reshape(-1).to(torch.int64)
    expanded_weights = topk_weights.reshape(-1).to(torch.float32)

    sort_idx = torch.argsort(expanded_expert_ids)

    sorted_expert_ids = expanded_expert_ids[sort_idx]
    sorted_token_ids = expanded_token_ids[sort_idx].contiguous()
    sorted_token_weights = expanded_weights[sort_idx].contiguous()

    x_sorted = x[sorted_token_ids].contiguous()

    counts = torch.bincount(sorted_expert_ids, minlength=num_experts)

    expert_offsets = torch.empty(
        (num_experts + 1,),
        device=x.device,
        dtype=torch.int64,
    )
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(counts, dim=0)

    return x_sorted, sorted_token_ids, sorted_token_weights, expert_offsets.contiguous(), counts


def prepare_case(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    routing: str,
    dtype=torch.float16,
    device="cuda",
):
    topk_ids, topk_weights = make_topk_assignments(
        num_tokens,
        num_experts,
        top_k,
        routing,
        device=device,
    )

    x, w13, w2 = init_moe_tensors(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=dtype,
        device=device,
    )

    x_sorted, sorted_token_ids, sorted_token_weights, expert_offsets, counts = (
        build_sorted_moe_inputs(x, topk_ids, topk_weights, num_experts)
    )

    return {
        "x": x,
        "w13": w13,
        "w2": w2,
        "x_sorted": x_sorted,
        "sorted_token_ids": sorted_token_ids,
        "sorted_token_weights": sorted_token_weights,
        "expert_offsets": expert_offsets,
        "counts": counts,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
        "routing": routing,
        "dtype": dtype,
    }


# ----------------------------------------------------------------------
# Reference and validation
# ----------------------------------------------------------------------

def torch_fused_moe_ref(case, output):
    x_sorted = case["x_sorted"]
    w13 = case["w13"]
    w2 = case["w2"]
    expert_offsets = case["expert_offsets"]
    sorted_token_ids = case["sorted_token_ids"]
    sorted_token_weights = case["sorted_token_weights"]

    E, H, two_I = w13.shape
    _, I, _ = w2.shape

    output.zero_()

    for e in range(E):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if end <= start:
            continue

        x_e = x_sorted[start:end]
        token_ids = sorted_token_ids[start:end]
        weights = sorted_token_weights[start:end]

        tmp13 = torch.mm(x_e, w13[e])
        gate = tmp13[:, :I]
        up = tmp13[:, I:]

        hidden = (F.silu(gate.float()) * up.float()).to(x_sorted.dtype)
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


def build_metadata(case, pair):
    H = case["hidden_size"]
    I = case["intermediate_size"]
    expert_offsets = case["expert_offsets"]

    g1 = pair["g1"]
    g2 = pair["g2"]

    gemm1_metadata = fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        I,
        BLOCK_M=g1["BLOCK_M"],
        BLOCK_N=g1["BLOCK_N"],
        GROUP_M=g1["GROUP_M"],
    )

    gemm2_metadata = fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        H,
        BLOCK_M=g2["BLOCK_M"],
        BLOCK_N=g2["BLOCK_N"],
        GROUP_M=g2["GROUP_M"],
    )

    return gemm1_metadata, gemm2_metadata


def run_triton_fused_moe(
    case,
    pair,
    output,
    metadata,
    hidden_workspace=None,
    persistent_gemm2: bool = False,
    persistent_waves: int = 2,
):
    gemm1_metadata, gemm2_metadata = metadata

    use_atomic = case["top_k"] != 1
    clear_output = use_atomic

    fray.triton.fused_moe(
        case["x_sorted"],
        case["w13"],
        case["w2"],
        case["expert_offsets"],
        case["sorted_token_ids"],
        case["sorted_token_weights"],
        output=output,
        num_tokens=case["num_tokens"],
        gemm1_cfg=pair["g1"],
        gemm2_cfg=pair["g2"],
        gemm1_metadata=gemm1_metadata,
        gemm2_metadata=gemm2_metadata,
        use_atomic=use_atomic,
        clear_output=clear_output,
        persistent_gemm2=persistent_gemm2,
        persistent_waves=persistent_waves,
        hidden=hidden_workspace,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_distribution(case):
    counts = case["counts"].detach().cpu().tolist()
    active = sum(1 for x in counts if x > 0)

    print(f"tokens       : {case['num_tokens']}")
    print(f"top_k        : {case['top_k']}")
    print(f"expanded     : {case['num_tokens'] * case['top_k']}")
    print(f"experts      : {case['num_experts']}")
    print(f"active       : {active}")
    print(f"max/expert   : {max(counts) if counts else 0}")
    print(f"counts sample: {counts[:min(16, len(counts))]}")


def case_title(case):
    return (
        f"N={case['num_tokens']}, E={case['num_experts']}, "
        f"H={case['hidden_size']}, I={case['intermediate_size']}, "
        f"top_k={case['top_k']}, routing={case['routing']}"
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 80)
    print("Accuracy Test: fused_moe")
    print("=" * 80)

    cases = [
        (32, 4, 128, 256, 1, "uniform"),
        (64, 4, 128, 256, 2, "uniform"),
        (127, 8, 255, 513, 2, "with_empty"),
        (256, 16, 1024, 2816, 2, "skewed"),
    ]

    for args in cases:
        case = prepare_case(*args)
        pair = DEFAULT_PAIR

        out = torch.empty(
            (case["num_tokens"], case["hidden_size"]),
            device="cuda",
            dtype=case["dtype"],
        )

        ref = torch.empty_like(out)

        metadata = build_metadata(case, pair)

        run_triton_fused_moe(case, pair, out, metadata)
        torch_fused_moe_ref(case, ref)

        stats = compare_outputs(out, ref)

        print("\n" + case_title(case))
        print_distribution(case)
        print(f"config       : {pair['name']}")
        print(f"sample out   : {out[0, :4].tolist()}")
        print(f"sample ref   : {ref[0, :4].tolist()}")
        print(f"accuracy     : {'PASS' if stats['ok'] else 'FAIL'}")
        print(f"max diff     : {stats['max_diff']:.6f}")
        print(f"mean diff    : {stats['mean_diff']:.6f}")
        print(f"nonfinite out: {stats['nonfinite_out']}")
        print(f"nonfinite ref: {stats['nonfinite_ref']}")


def perf_test():
    torch.manual_seed(42)

    print("\n" + "=" * 80)
    print("Performance Test: fused_moe vs torch reference")
    print("=" * 80)

    cases = [
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "skewed"),
    ]

    for args in cases:
        case = prepare_case(*args)
        pair = DEFAULT_PAIR

        out = torch.empty(
            (case["num_tokens"], case["hidden_size"]),
            device="cuda",
            dtype=case["dtype"],
        )

        hidden_workspace = torch.empty(
            (
                case["num_tokens"] * case["top_k"],
                case["intermediate_size"],
            ),
            device="cuda",
            dtype=case["dtype"],
            )

        ref = torch.empty_like(out)

        metadata = build_metadata(case, pair)

        def run_triton():
            run_triton_fused_moe(case, pair, out, metadata, hidden_workspace)

        def run_ref():
            torch_fused_moe_ref(case, ref)

        run_triton()
        run_ref()
        torch.cuda.synchronize()

        stats = compare_outputs(out, ref)

        t_triton = bench_kineto(run_triton, "triton_fused_moe")
        t_ref = bench_kineto(run_ref, "torch_fused_moe_ref")

        T = case["num_tokens"] * case["top_k"]
        H = case["hidden_size"]
        I = case["intermediate_size"]
        flops = 6.0 * T * H * I

        print("\n" + case_title(case))
        print_distribution(case)
        print(f"config       : {pair['name']}")
        print(f"accuracy     : {'PASS' if stats['ok'] else 'FAIL'}")
        print(f"triton       : {t_triton * 1e6:8.2f} us | {flops / t_triton / 1e12:8.2f} TFLOPS")
        print(f"torch ref    : {t_ref * 1e6:8.2f} us | {flops / t_ref / 1e12:8.2f} TFLOPS")
        print(f"speedup      : {t_ref / t_triton:8.2f}x")
        print(f"max diff     : {stats['max_diff']:.6f}")
        print(f"mean diff    : {stats['mean_diff']:.6f}")


def robust_config_sweep():
    """
    多 shape 汇总。

    目标不是找每个 shape 的单点最优，而是找一个鲁棒默认配置：
        ratio = 当前配置耗时 / 当前 shape 最快耗时

    重点看：
        avg_ratio
        worst_ratio
        wins
        failures
    """

    torch.manual_seed(42)

    print("\n" + "=" * 80)
    print("Robust Config Sweep")
    print("=" * 80)

    cases = [
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (512, 8, 1024, 2816, 2, "with_empty"),
        (1024, 16, 2048, 5632, 2, "uniform"),
        (1024, 16, 2048, 5632, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "with_empty"),
    ]

    summary = {
        pair["name"]: {
            "ratios": [],
            "wins": 0,
            "failures": 0,
            "times": [],
        }
        for pair in CANDIDATE_PAIRS
    }

    for args in cases:
        case = prepare_case(*args)

        ref = torch.empty(
            (case["num_tokens"], case["hidden_size"]),
            device="cuda",
            dtype=case["dtype"],
        )
        torch_fused_moe_ref(case, ref)
        torch.cuda.synchronize()

        print("\n" + "-" * 80)
        print(case_title(case))
        print_distribution(case)

        case_results = []

        for pair in CANDIDATE_PAIRS:
            out = torch.empty_like(ref)

            try:
                metadata = build_metadata(case, pair)

                def run_triton():
                    run_triton_fused_moe(case, pair, out, metadata)

                run_triton()
                torch.cuda.synchronize()

                stats = compare_outputs(out, ref)

                if not stats["ok"]:
                    raise RuntimeError(
                        f"accuracy failed: max={stats['max_diff']:.6f}, mean={stats['mean_diff']:.6f}"
                    )

                t = bench_kineto(run_triton, f"fused_moe_{pair['name']}")
                case_results.append((pair["name"], t, stats, metadata))

            except Exception as e:
                summary[pair["name"]]["failures"] += 1
                print(f"{pair['name']:>12}: FAIL | {type(e).__name__}: {str(e).splitlines()[0][:120]}")

        if not case_results:
            print("No valid configs for this case.")
            continue

        best_time = min(x[1] for x in case_results)
        best_name = min(case_results, key=lambda x: x[1])[0]

        for name, t, stats, metadata in case_results:
            ratio = t / best_time

            summary[name]["ratios"].append(ratio)
            summary[name]["times"].append(t)

            if name == best_name:
                summary[name]["wins"] += 1

            g1_tiles = metadata[0][0].numel()
            g2_tiles = metadata[1][0].numel()

            print(
                f"{name:>12}: "
                f"{t * 1e6:8.2f} us | "
                f"ratio={ratio:5.3f} | "
                f"G1_tiles={g1_tiles:5d} | "
                f"G2_tiles={g2_tiles:5d} | "
                f"max_diff={stats['max_diff']:.6f}"
            )

    print("\n" + "=" * 80)
    print("Robust Config Summary")
    print("=" * 80)

    rows = []

    for pair in CANDIDATE_PAIRS:
        name = pair["name"]
        item = summary[name]
        ratios = item["ratios"]

        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            worst_ratio = max(ratios)
            avg_time_us = sum(item["times"]) / len(item["times"]) * 1e6
        else:
            avg_ratio = float("inf")
            worst_ratio = float("inf")
            avg_time_us = float("inf")

        rows.append(
            (
                avg_ratio,
                worst_ratio,
                -item["wins"],
                item["failures"],
                avg_time_us,
                name,
            )
        )

    rows.sort()

    print(
        f"{'config':>12} | "
        f"{'avg_ratio':>9} | "
        f"{'worst':>9} | "
        f"{'wins':>4} | "
        f"{'fail':>4} | "
        f"{'avg_us':>10}"
    )
    print("-" * 64)

    for avg_ratio, worst_ratio, neg_wins, failures, avg_time_us, name in rows:
        print(
            f"{name:>12} | "
            f"{avg_ratio:9.3f} | "
            f"{worst_ratio:9.3f} | "
            f"{-neg_wins:4d} | "
            f"{failures:4d} | "
            f"{avg_time_us:10.2f}"
        )

    best_name = rows[0][-1]
    print("\nRecommended robust default config:", best_name)

    for pair in CANDIDATE_PAIRS:
        if pair["name"] == best_name:
            print("GEMM1:")
            print(pair["g1"])
            print("GEMM2:")
            print(pair["g2"])
            break

    print("=" * 80 + "\n")

def persistent_gemm2_test():
    torch.manual_seed(42)

    print("\n" + "=" * 80)
    print("Persistent GEMM2 Test")
    print("=" * 80)

    cases = [
        (512, 8, 1024, 2816, 2, "uniform"),
        (512, 8, 1024, 2816, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "uniform"),
        (1024, 16, 2048, 5632, 2, "skewed"),
        (1024, 16, 2048, 5632, 2, "with_empty"),
    ]

    pair = DEFAULT_PAIR

    for args in cases:
        case = prepare_case(*args)

        out_normal = torch.empty(
            (case["num_tokens"], case["hidden_size"]),
            device="cuda",
            dtype=case["dtype"],
        )

        out_persistent = torch.empty_like(out_normal)
        ref = torch.empty_like(out_normal)

        metadata = build_metadata(case, pair)

        torch_fused_moe_ref(case, ref)

        def run_normal():
            run_triton_fused_moe(
                case,
                pair,
                out_normal,
                metadata,
                persistent_gemm2=False,
            )

        def run_persistent():
            run_triton_fused_moe(
                case,
                pair,
                out_persistent,
                metadata,
                persistent_gemm2=True,
                persistent_waves=16,
            )

        run_normal()
        run_persistent()
        torch.cuda.synchronize()

        normal_stats = compare_outputs(out_normal, ref)
        persistent_stats = compare_outputs(out_persistent, ref)

        t_normal = bench_kineto(run_normal, "fused_moe_normal_gemm2")
        t_persistent = bench_kineto(run_persistent, "fused_moe_persistent_gemm2")

        print("\n" + case_title(case))
        print_distribution(case)
        print(f"normal accuracy     : {'PASS' if normal_stats['ok'] else 'FAIL'}")
        print(f"persistent accuracy : {'PASS' if persistent_stats['ok'] else 'FAIL'}")
        print(f"normal time         : {t_normal * 1e6:8.2f} us")
        print(f"persistent time     : {t_persistent * 1e6:8.2f} us")
        print(f"persistent speedup  : {t_normal / t_persistent:8.2f}x")
        print(f"persistent max diff : {persistent_stats['max_diff']:.6f}")
        print(f"persistent mean diff: {persistent_stats['mean_diff']:.6f}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    # accuracy_test()
    perf_test()
    # robust_config_sweep()
    # persistent_gemm2_test()