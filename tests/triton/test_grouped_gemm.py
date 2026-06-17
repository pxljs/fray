import torch

import fray
from fray import bench_kineto


RTOL = 1e-2
ATOL = 5e-1


GROUPED_GEMM_CONFIGS = [
    {
        "name": "BM64_BN128_BK32_GM8_W4_S4",
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "BM64_BN128_BK64_GM8_W4_S4",
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "BM128_BN128_BK32_GM8_W4_S4",
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "BM128_BN128_BK64_GM8_W4_S3",
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "name": "BM64_BN256_BK32_GM8_W8_S4",
        "BLOCK_M": 64,
        "BLOCK_N": 256,
        "BLOCK_K": 32,
        "GROUP_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    },
    {
        "name": "BM64_BN256_BK64_GM8_W8_S3",
        "BLOCK_M": 64,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "name": "BM128_BN256_BK64_GM8_W8_S3",
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "name": "BM32_BN128_BK64_GM8_W4_S4",
        "BLOCK_M": 32,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "BM64_BN128_BK64_GM4_W4_S4",
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 4,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "name": "BM64_BN128_BK64_GM16_W4_S4",
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 16,
        "num_warps": 4,
        "num_stages": 4,
    },
]


DEFAULT_CFG = GROUPED_GEMM_CONFIGS[1]


def make_expert_counts(
    total_tokens: int,
    num_experts: int,
    mode: str,
):
    assert total_tokens >= 0
    assert num_experts > 0

    if mode == "uniform":
        base = total_tokens // num_experts
        rem = total_tokens % num_experts

        counts = [base] * num_experts
        for i in range(rem):
            counts[i] += 1

        return counts

    if mode == "skewed":
        counts = [0] * num_experts

        hot0 = min(total_tokens, int(total_tokens * 0.50))
        hot1 = min(total_tokens - hot0, int(total_tokens * 0.25))
        hot2 = min(total_tokens - hot0 - hot1, int(total_tokens * 0.10))

        if num_experts >= 1:
            counts[0] = hot0
        if num_experts >= 2:
            counts[1] = hot1
        if num_experts >= 3:
            counts[2] = hot2

        remaining = total_tokens - sum(counts)

        start = min(3, num_experts)
        active_tail = max(start + 1, num_experts // 2)

        for i in range(remaining):
            e = start + (i % max(1, active_tail - start))
            if e < num_experts:
                counts[e] += 1
            else:
                counts[num_experts - 1] += 1

        assert sum(counts) == total_tokens
        return counts

    if mode == "with_empty":
        counts = [0] * num_experts

        active = max(1, num_experts // 2)
        base = total_tokens // active
        rem = total_tokens % active

        for i in range(active):
            counts[i] = base + (1 if i < rem else 0)

        assert sum(counts) == total_tokens
        return counts

    if mode == "single_hot":
        counts = [0] * num_experts
        counts[0] = total_tokens
        return counts

    raise ValueError(f"Unknown mode: {mode}")


def counts_to_offsets(
    counts,
    device="cuda",
    dtype=torch.int64,
):
    offsets = [0]

    cur = 0
    for c in counts:
        cur += int(c)
        offsets.append(cur)

    return torch.tensor(
        offsets,
        device=device,
        dtype=dtype,
    )


def torch_grouped_gemm_ref(
    x: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    output: torch.Tensor,
):
    num_experts = weights.shape[0]

    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())

        if end <= start:
            continue

        torch.mm(
            x[start:end, :],
            weights[e],
            out=output[start:end, :],
        )

    return output


def compare_outputs(
    out_triton: torch.Tensor,
    out_ref: torch.Tensor,
    with_quantile: bool = True,
):
    torch.cuda.synchronize()

    diff = torch.abs(out_triton.float() - out_ref.float())

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    p99_diff = None
    p999_diff = None

    if with_quantile:
        flat = diff.flatten()

        max_quantile_elements = 1_000_000

        if flat.numel() > max_quantile_elements:
            idx = torch.randint(
                0,
                flat.numel(),
                (max_quantile_elements,),
                device=flat.device,
            )
            q_input = flat[idx]
        else:
            q_input = flat

        p99_diff = torch.quantile(q_input, 0.99).item()
        p999_diff = torch.quantile(q_input, 0.999).item()

    is_close = torch.allclose(
        out_triton,
        out_ref,
        rtol=RTOL,
        atol=ATOL,
    )

    return {
        "is_close": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "p99_diff": p99_diff,
        "p999_diff": p999_diff,
    }


def print_compare_result(stats):
    print(f"Accuracy Check      : {'✅ PASS' if stats['is_close'] else '❌ FAIL'}")
    print(f"Max Diff            : {stats['max_diff']:.6f}")
    print(f"Mean Diff           : {stats['mean_diff']:.6f}")

    if stats["p99_diff"] is not None:
        print(f"P99 Diff            : {stats['p99_diff']:.6f}")

    if stats["p999_diff"] is not None:
        print(f"P999 Diff           : {stats['p999_diff']:.6f}")


def print_expert_distribution(counts):
    total = sum(counts)
    active = sum(1 for x in counts if x > 0)
    max_count = max(counts) if counts else 0
    min_nonzero = min([x for x in counts if x > 0], default=0)

    print(f"Total tokens        : {total}")
    print(f"Num experts         : {len(counts)}")
    print(f"Active experts      : {active}")
    print(f"Max tokens/expert   : {max_count}")
    print(f"Min nonzero/expert  : {min_nonzero}")
    print(f"Counts sample       : {counts[:min(16, len(counts))]}")


def build_metadata_for_cfg(
    expert_offsets: torch.Tensor,
    N: int,
    cfg: dict,
):
    return fray.triton.build_grouped_gemm_metadata(
        expert_offsets,
        N,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        GROUP_M=cfg["GROUP_M"],
    )


def run_grouped_gemm_for_cfg(
    x: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_expert_ids: torch.Tensor,
    tile_m_ids: torch.Tensor,
    tile_n_ids: torch.Tensor,
    cfg: dict,
):
    return fray.triton.grouped_gemm(
        x,
        weights,
        output,
        expert_offsets,
        tile_expert_ids,
        tile_m_ids,
        tile_n_ids,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def run_grouped_gemm_once(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    distribution: str,
    cfg: dict = DEFAULT_CFG,
    dtype=torch.float16,
    device="cuda",
    check_quantile: bool = True,
):
    counts = make_expert_counts(
        total_tokens=total_tokens,
        num_experts=num_experts,
        mode=distribution,
    )

    expert_offsets = counts_to_offsets(
        counts,
        device=device,
        dtype=torch.int64,
    )

    x = torch.randn(
        (total_tokens, K),
        device=device,
        dtype=dtype,
    )

    weights = torch.randn(
        (num_experts, K, N),
        device=device,
        dtype=dtype,
    )

    out_triton = torch.empty(
        (total_tokens, N),
        device=device,
        dtype=dtype,
    )

    out_ref = torch.empty_like(out_triton)

    tile_expert_ids, tile_m_ids, tile_n_ids = build_metadata_for_cfg(
        expert_offsets,
        N,
        cfg,
    )

    run_grouped_gemm_for_cfg(
        x,
        weights,
        out_triton,
        expert_offsets,
        tile_expert_ids,
        tile_m_ids,
        tile_n_ids,
        cfg,
    )

    torch_grouped_gemm_ref(
        x,
        weights,
        expert_offsets,
        out_ref,
    )

    stats = compare_outputs(
        out_triton,
        out_ref,
        with_quantile=check_quantile,
    )

    return {
        "x": x,
        "weights": weights,
        "out_triton": out_triton,
        "out_ref": out_ref,
        "expert_offsets": expert_offsets,
        "tile_expert_ids": tile_expert_ids,
        "tile_m_ids": tile_m_ids,
        "tile_n_ids": tile_n_ids,
        "counts": counts,
        "stats": stats,
    }


def grouped_gemm_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton Grouped GEMM vs Python loop torch.mm")
    print("=" * 60)
    print(f"Using default config: {DEFAULT_CFG['name']}")

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (256, 4, 128, 128, "uniform"),
        (512, 8, 256, 256, "uniform"),
        (1024, 8, 512, 512, "skewed"),
        (2048, 16, 1024, 512, "with_empty"),
        (2048, 16, 512, 1024, "skewed"),
        (4096, 32, 1024, 1024, "with_empty"),
        (4096, 32, 4096, 1024, "skewed"),
        (2048, 16, 4096, 11008, "uniform"),
        (2048, 16, 11008, 4096, "skewed"),
    ]

    for total_tokens, num_experts, K, N, distribution in test_cases:
        print(
            f"\nConfiguration: total_tokens={total_tokens}, "
            f"E={num_experts}, K={K}, N={N}, "
            f"distribution={distribution}, dtype={dtype}"
        )

        result = run_grouped_gemm_once(
            total_tokens=total_tokens,
            num_experts=num_experts,
            K=K,
            N=N,
            distribution=distribution,
            cfg=DEFAULT_CFG,
            dtype=dtype,
            device=device,
            check_quantile=True,
        )

        print_expert_distribution(result["counts"])
        print(f"Total grouped tiles  : {result['tile_expert_ids'].numel()}")

        out_triton = result["out_triton"]
        out_ref = result["out_ref"]
        stats = result["stats"]

        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print_compare_result(stats)

    print("=" * 60 + "\n")


def grouped_gemm_edge_case_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Edge Case Test: empty experts / single hot expert / non-multiple shapes")
    print("=" * 60)
    print(f"Using default config: {DEFAULT_CFG['name']}")

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (127, 8, 255, 129, "uniform"),
        (513, 16, 1025, 257, "with_empty"),
        (1024, 16, 4097, 1001, "skewed"),
        (2048, 32, 4096, 513, "single_hot"),
    ]

    for total_tokens, num_experts, K, N, distribution in test_cases:
        print(
            f"\nConfiguration: total_tokens={total_tokens}, "
            f"E={num_experts}, K={K}, N={N}, "
            f"distribution={distribution}, dtype={dtype}"
        )

        result = run_grouped_gemm_once(
            total_tokens=total_tokens,
            num_experts=num_experts,
            K=K,
            N=N,
            distribution=distribution,
            cfg=DEFAULT_CFG,
            dtype=dtype,
            device=device,
            check_quantile=True,
        )

        print_expert_distribution(result["counts"])
        print(f"Total grouped tiles  : {result['tile_expert_ids'].numel()}")

        out_triton = result["out_triton"]
        out_ref = result["out_ref"]
        stats = result["stats"]

        print(f"Triton Output Sample: {out_triton[0, :4].tolist()}")
        print(f"Torch Ref Sample    : {out_ref[0, :4].tolist()}")
        print_compare_result(stats)

    print("=" * 60 + "\n")


def test_grouped_gemm_performance():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton Grouped GEMM vs Python loop torch.mm")
    print("=" * 60)
    print(f"Using default config: {DEFAULT_CFG['name']}")

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (2048, 8, 1024, 1024, "uniform"),
        (2048, 8, 1024, 1024, "skewed"),
        (4096, 16, 2048, 2048, "uniform"),
        (4096, 16, 2048, 2048, "skewed"),
        (4096, 32, 4096, 1024, "with_empty"),
        (4096, 32, 1024, 4096, "skewed"),

        # MoE FFN-like
        (2048, 16, 4096, 11008, "uniform"),
        (2048, 16, 4096, 11008, "skewed"),
        (2048, 16, 11008, 4096, "uniform"),
        (2048, 16, 11008, 4096, "skewed"),
    ]

    for total_tokens, num_experts, K, N, distribution in test_cases:
        print(
            f"\nConfiguration: total_tokens={total_tokens}, "
            f"E={num_experts}, K={K}, N={N}, "
            f"distribution={distribution}, dtype={dtype}"
        )

        counts = make_expert_counts(
            total_tokens=total_tokens,
            num_experts=num_experts,
            mode=distribution,
        )

        expert_offsets = counts_to_offsets(
            counts,
            device=device,
            dtype=torch.int64,
        )

        x = torch.randn(
            (total_tokens, K),
            device=device,
            dtype=dtype,
        )

        weights = torch.randn(
            (num_experts, K, N),
            device=device,
            dtype=dtype,
        )

        out_triton = torch.empty(
            (total_tokens, N),
            device=device,
            dtype=dtype,
        )

        out_torch = torch.empty_like(out_triton)

        tile_expert_ids, tile_m_ids, tile_n_ids = build_metadata_for_cfg(
            expert_offsets,
            N,
            DEFAULT_CFG,
        )

        print_expert_distribution(counts)
        print(f"Total grouped tiles  : {tile_expert_ids.numel()}")

        def run_triton():
            run_grouped_gemm_for_cfg(
                x,
                weights,
                out_triton,
                expert_offsets,
                tile_expert_ids,
                tile_m_ids,
                tile_n_ids,
                DEFAULT_CFG,
            )

        def run_torch_loop():
            torch_grouped_gemm_ref(
                x,
                weights,
                expert_offsets,
                out_torch,
            )

        run_triton()
        run_torch_loop()
        torch.cuda.synchronize()

        stats = compare_outputs(
            out_triton,
            out_torch,
            with_quantile=False,
        )

        if not stats["is_close"]:
            print(
                f"WARNING: correctness mismatch, "
                f"max_diff={stats['max_diff']:.6f}, "
                f"mean_diff={stats['mean_diff']:.6f}"
            )

        t_torch_s = bench_kineto(
            run_torch_loop,
            "torch_grouped_gemm_loop",
        )

        t_triton_s = bench_kineto(
            run_triton,
            "triton_grouped_gemm",
        )

        total_flops = 2.0 * total_tokens * K * N

        triton_tflops = total_flops / t_triton_s / 1e12
        torch_tflops = total_flops / t_torch_s / 1e12

        total_bytes = (
            total_tokens * K * x.element_size()
            + num_experts * K * N * weights.element_size()
            + total_tokens * N * out_triton.element_size()
        )

        triton_gbps = total_bytes / t_triton_s / 1e9
        torch_gbps = total_bytes / t_torch_s / 1e9

        print("-" * 60)
        print(
            f"Triton Grouped GEMM : {t_triton_s * 1e6:8.2f} us | "
            f"Compute: {triton_tflops:8.2f} TFLOPS | "
            f"BW: {triton_gbps:8.2f} GB/s"
        )
        print(
            f"Torch loop torch.mm : {t_torch_s * 1e6:8.2f} us | "
            f"Compute: {torch_tflops:8.2f} TFLOPS | "
            f"BW: {torch_gbps:8.2f} GB/s"
        )

        if t_triton_s > 0:
            print(f"Speedup             : {t_torch_s / t_triton_s:8.2f}x")

        print(f"Max Diff            : {stats['max_diff']:.6f}")
        print(f"Mean Diff           : {stats['mean_diff']:.6f}")
        print("-" * 60)


def sweep_grouped_gemm_configs(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    distribution: str,
    dtype=torch.float16,
    device="cuda",
):
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Grouped GEMM Config Sweep")
    print("=" * 60)

    counts = make_expert_counts(
        total_tokens=total_tokens,
        num_experts=num_experts,
        mode=distribution,
    )

    expert_offsets = counts_to_offsets(
        counts,
        device=device,
        dtype=torch.int64,
    )

    x = torch.randn(
        (total_tokens, K),
        device=device,
        dtype=dtype,
    )

    weights = torch.randn(
        (num_experts, K, N),
        device=device,
        dtype=dtype,
    )

    out_triton = torch.empty(
        (total_tokens, N),
        device=device,
        dtype=dtype,
    )

    out_ref = torch.empty_like(out_triton)

    torch_grouped_gemm_ref(
        x,
        weights,
        expert_offsets,
        out_ref,
    )

    torch.cuda.synchronize()

    print(
        f"\nConfiguration: total_tokens={total_tokens}, "
        f"E={num_experts}, K={K}, N={N}, "
        f"distribution={distribution}, dtype={dtype}"
    )

    print_expert_distribution(counts)

    total_flops = 2.0 * total_tokens * K * N

    best = None

    for cfg in GROUPED_GEMM_CONFIGS:
        print("\n" + "-" * 60)
        print(f"Testing config: {cfg['name']}")

        tile_expert_ids, tile_m_ids, tile_n_ids = build_metadata_for_cfg(
            expert_offsets,
            N,
            cfg,
        )

        print(f"Total grouped tiles : {tile_expert_ids.numel()}")

        def run_triton():
            run_grouped_gemm_for_cfg(
                x,
                weights,
                out_triton,
                expert_offsets,
                tile_expert_ids,
                tile_m_ids,
                tile_n_ids,
                cfg,
            )

        run_triton()
        torch.cuda.synchronize()

        stats = compare_outputs(
            out_triton,
            out_ref,
            with_quantile=False,
        )

        if not stats["is_close"]:
            print(
                f"Accuracy: ❌ FAIL | "
                f"max_diff={stats['max_diff']:.6f}, "
                f"mean_diff={stats['mean_diff']:.6f}"
            )
            continue

        t_s = bench_kineto(
            run_triton,
            f"triton_grouped_gemm_{cfg['name']}",
        )

        tflops = total_flops / t_s / 1e12

        print(
            f"Accuracy: ✅ PASS | "
            f"Time: {t_s * 1e6:8.2f} us | "
            f"TFLOPS: {tflops:8.2f} | "
            f"MaxDiff: {stats['max_diff']:.6f} | "
            f"MeanDiff: {stats['mean_diff']:.6f}"
        )

        if best is None or t_s < best["time"]:
            best = {
                "config": cfg,
                "time": t_s,
                "tflops": tflops,
                "max_diff": stats["max_diff"],
                "mean_diff": stats["mean_diff"],
                "tiles": tile_expert_ids.numel(),
            }

    print("\n" + "=" * 60)
    if best is not None:
        print("Best config:")
        print(best["config"])
        print(f"Best time          : {best['time'] * 1e6:.2f} us")
        print(f"Best TFLOPS        : {best['tflops']:.2f}")
        print(f"Total grouped tiles: {best['tiles']}")
        print(f"Max Diff           : {best['max_diff']:.6f}")
        print(f"Mean Diff          : {best['mean_diff']:.6f}")
    else:
        print("No valid config found.")
    print("=" * 60 + "\n")


def test_metadata_build_cost():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Metadata Build Cost Test")
    print("=" * 60)

    device = "cuda"

    test_cases = [
        (2048, 8, 1024, "uniform"),
        (4096, 16, 2048, "skewed"),
        (8192, 32, 4096, "with_empty"),
        (8192, 64, 4096, "skewed"),
    ]

    for total_tokens, num_experts, N, distribution in test_cases:
        print(
            f"\nConfiguration: total_tokens={total_tokens}, "
            f"E={num_experts}, N={N}, distribution={distribution}"
        )

        counts = make_expert_counts(
            total_tokens=total_tokens,
            num_experts=num_experts,
            mode=distribution,
        )

        expert_offsets = counts_to_offsets(
            counts,
            device=device,
            dtype=torch.int64,
        )

        def run_metadata():
            build_metadata_for_cfg(
                expert_offsets,
                N,
                DEFAULT_CFG,
            )

        run_metadata()
        torch.cuda.synchronize()

        t_meta_s = bench_kineto(
            run_metadata,
            "build_grouped_gemm_metadata",
        )

        tile_expert_ids, tile_m_ids, tile_n_ids = build_metadata_for_cfg(
            expert_offsets,
            N,
            DEFAULT_CFG,
        )

        print_expert_distribution(counts)
        print(f"Total grouped tiles  : {tile_expert_ids.numel()}")
        print(f"Metadata build time  : {t_meta_s * 1e6:8.2f} us")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    grouped_gemm_accuracy_test()

    print("\nStep 2: Running Edge Case Test...")
    grouped_gemm_edge_case_test()

    print("\nStep 3: Running Performance Test...")
    test_grouped_gemm_performance()

    print("\nStep 4: Running Metadata Build Cost Test...")
    test_metadata_build_cost()

    print("\nStep 5: Running Config Sweep for slow down-proj case...")
    sweep_grouped_gemm_configs(
        total_tokens=2048,
        num_experts=16,
        K=11008,
        N=4096,
        distribution="skewed",
    )

    print("\nStep 6: Running Config Sweep for up/gate-proj case...")
    sweep_grouped_gemm_configs(
        total_tokens=2048,
        num_experts=16,
        K=4096,
        N=11008,
        distribution="skewed",
    )