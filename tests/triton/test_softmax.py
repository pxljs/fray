import math
import torch

import fray
from fray import bench_kineto


def torch_softmax_ref(
    x: torch.Tensor,
    scale: float = 1.0,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    y = x.float() * scale

    if mask is not None:
        if mask.dtype == torch.bool:
            y = y.masked_fill(~mask, float("-inf"))
        else:
            y = y + mask.float()

    if is_causal:
        q_len = x.shape[-2]
        kv_len = x.shape[-1]

        causal_mask = torch.ones(
            (q_len, kv_len),
            device=x.device,
            dtype=torch.bool,
        ).tril()

        y = y.masked_fill(~causal_mask, float("-inf"))

    out = torch.softmax(y, dim=-1)

    return out.to(x.dtype)


def make_bool_mask(shape, device):
    # True 表示保留，False 表示 mask 掉
    mask = torch.rand(shape, device=device) > 0.25

    # 避免某一行全 False，否则 PyTorch softmax(-inf, -inf, ...)
    # 会产生 NaN，而我们的 Triton kernel 通常会输出全 0。
    mask[..., 0] = True

    return mask


def make_additive_mask(shape, device, dtype):
    # additive mask:
    #   0      表示保留
    #   -10000 表示屏蔽
    #
    # 这里不用 -inf，是为了避免某些情况下 PyTorch baseline 出 NaN。
    keep = torch.rand(shape, device=device) > 0.25
    keep[..., 0] = True

    mask = torch.zeros(shape, device=device, dtype=dtype)
    mask = mask.masked_fill(~keep, -10000.0)

    return mask


def run_one_accuracy_case(
    x: torch.Tensor,
    mode_name: str,
    scale: float = 1.0,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    out_triton = torch.empty_like(x)

    fray.triton.softmax(
        x,
        out_triton,
        scale=scale,
        mask=mask,
        is_causal=is_causal,
    )

    out_ref = torch_softmax_ref(
        x,
        scale=scale,
        mask=mask,
        is_causal=is_causal,
    )

    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(out_triton - out_ref)).item()
    mean_diff = torch.mean(torch.abs(out_triton - out_ref)).item()

    is_close = torch.allclose(out_triton, out_ref, rtol=1e-2, atol=1e-2)

    row_sum_triton = out_triton.reshape(-1, x.shape[-1])[0].sum().item()
    row_sum_ref = out_ref.reshape(-1, x.shape[-1])[0].sum().item()

    print(f"\nMode = {mode_name}, shape={list(x.shape)}, dtype={x.dtype}")
    print(f"Triton Output Sample: {out_triton.reshape(-1, x.shape[-1])[0, :4].tolist()}")
    print(f"Torch Ref Sample    : {out_ref.reshape(-1, x.shape[-1])[0, :4].tolist()}")
    print(f"Accuracy Check      : {'✅ PASS' if is_close else '❌ FAIL'}")
    print(f"Max Diff            : {max_diff:.6f}")
    print(f"Mean Diff           : {mean_diff:.6f}")
    print(f"Row Sum Triton      : {row_sum_triton:.6f}")
    print(f"Row Sum Torch       : {row_sum_ref:.6f}")


def softmax_accuracy_test():
    torch.manual_seed(42)

    print("\n" + "=" * 60)
    print("Accuracy Test: Triton Softmax vs PyTorch torch.softmax")
    print("=" * 60)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (1, 128),
        (1, 1024),
        (8, 1024),
        (128, 1024),
        (128, 2048),
        (128, 4096),
        (128, 8192),
        (1024, 1024),
        (1024, 4096),
    ]

    for M, N in test_cases:
        x = torch.randn((M, N), device=device, dtype=dtype)

        scale = 1.0 / math.sqrt(N)
        bool_mask = make_bool_mask(x.shape, device=device)
        additive_mask = make_additive_mask(x.shape, device=device, dtype=dtype)

        run_one_accuracy_case(
            x,
            mode_name="softmax",
        )

        run_one_accuracy_case(
            x,
            mode_name="scale_softmax",
            scale=scale,
        )

        run_one_accuracy_case(
            x,
            mode_name="bool_mask_softmax",
            scale=scale,
            mask=bool_mask,
        )

        run_one_accuracy_case(
            x,
            mode_name="additive_mask_softmax",
            scale=scale,
            mask=additive_mask,
        )

        # 对 [M, N] 来说，这里会按照最后两维做 causal。
        # 如果 M != N，也可以理解成 q_len=M, kv_len=N。
        run_one_accuracy_case(
            x,
            mode_name="causal_softmax",
            scale=scale,
            is_causal=True,
        )

    print("=" * 60 + "\n")


def run_one_perf_case(
    x: torch.Tensor,
    mode_name: str,
    scale: float = 1.0,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
):
    out_triton = torch.empty_like(x)
    out_torch = torch.empty_like(x)

    def run_triton():
        fray.triton.softmax(
            x,
            out_triton,
            scale=scale,
            mask=mask,
            is_causal=is_causal,
        )

    def run_torch():
        out_torch.copy_(
            torch_softmax_ref(
                x,
                scale=scale,
                mask=mask,
                is_causal=is_causal,
            )
        )

    # 显式触发初始化 / Triton JIT，避免污染正式计时
    run_triton()
    run_torch()
    torch.cuda.synchronize()

    max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
    mean_diff = torch.mean(torch.abs(out_triton - out_torch)).item()
    is_close = torch.allclose(out_triton, out_torch, rtol=1e-2, atol=1e-2)

    if not is_close:
        print(
            f"WARNING: correctness mismatch, "
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    t_torch_s = bench_kineto(run_torch, f"torch_{mode_name}")
    t_triton_s = bench_kineto(run_triton, f"triton_{mode_name}")

    n_elements = x.numel()

    # softmax 逻辑有效访存量粗略按：
    # read x + write out
    #
    # 如果带 mask，实际还多读一次 mask。
    # PyTorch baseline 由于表达式拆分，也会有更多中间 tensor 读写。
    bytes_per_element = x.element_size() * 2

    if mask is not None:
        bytes_per_element += mask.element_size()

    total_bytes = n_elements * bytes_per_element

    triton_gbps = total_bytes / t_triton_s / 1e9
    torch_gbps = total_bytes / t_torch_s / 1e9

    print("-" * 60)
    print(f"Mode           : {mode_name}")
    print(f"Shape          : {list(x.shape)}, dtype={x.dtype}")
    print(
        f"Triton Softmax : {t_triton_s * 1e6:8.2f} us | "
        f"Effective Bandwidth: {triton_gbps:8.2f} GB/s"
    )
    print(
        f"PyTorch Softmax: {t_torch_s * 1e6:8.2f} us | "
        f"Effective Bandwidth: {torch_gbps:8.2f} GB/s"
    )

    if t_triton_s > 0:
        print(f"Speedup        : {t_torch_s / t_triton_s:8.2f}x")

    print(f"Max Diff       : {max_diff:.6f}")
    print(f"Mean Diff      : {mean_diff:.6f}")
    print("-" * 60)


def test_softmax_performance():
    print("\n" + "=" * 60)
    print("Performance Benchmark: Triton Softmax vs PyTorch torch.softmax")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (128, 128),
        (128, 256),
        (128, 512),
        (128, 1024),
        (128, 2048),
        (128, 4096),
        (128, 8192),
        (512, 1024),
        (1024, 1024),
        (1024, 2048),
        (1024, 4096),
        (2048, 2048),
        (4096, 1024),
        (4096, 2048),
    ]

    for M, N in test_cases:
        print(f"\nConfiguration: M={M}, N={N}, dtype={dtype}")

        x = torch.randn((M, N), device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(N)

        run_one_perf_case(
            x,
            mode_name="softmax",
        )

        run_one_perf_case(
            x,
            mode_name="scale_softmax",
            scale=scale,
        )

        # 性能测试里 mask case 会比较重，可以只测中等规模。
        if N <= 4096:
            bool_mask = make_bool_mask(x.shape, device=device)
            additive_mask = make_additive_mask(x.shape, device=device, dtype=dtype)

            run_one_perf_case(
                x,
                mode_name="bool_mask_softmax",
                scale=scale,
                mask=bool_mask,
            )

            run_one_perf_case(
                x,
                mode_name="additive_mask_softmax",
                scale=scale,
                mask=additive_mask,
            )

        run_one_perf_case(
            x,
            mode_name="causal_softmax",
            scale=scale,
            is_causal=True,
        )


def test_softmax_3d_shape():
    print("\n" + "=" * 60)
    print("3D Shape Test: [B, S, N]")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (1, 128, 1024),
        (2, 512, 1024),
        (4, 1024, 2048),
        (8, 1024, 4096),
    ]

    for B, S, N in test_cases:
        print(f"\nConfiguration: B={B}, S={S}, N={N}, dtype={dtype}")

        x = torch.randn((B, S, N), device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(N)

        bool_mask = make_bool_mask(x.shape, device=device)
        additive_mask = make_additive_mask(x.shape, device=device, dtype=dtype)

        run_one_perf_case(
            x,
            mode_name="3d_scale_softmax",
            scale=scale,
        )

        run_one_perf_case(
            x,
            mode_name="3d_bool_mask_softmax",
            scale=scale,
            mask=bool_mask,
        )

        run_one_perf_case(
            x,
            mode_name="3d_additive_mask_softmax",
            scale=scale,
            mask=additive_mask,
        )

        run_one_perf_case(
            x,
            mode_name="3d_causal_softmax",
            scale=scale,
            is_causal=True,
        )

    print("=" * 60 + "\n")


def test_softmax_attention_like_shape():
    print("\n" + "=" * 60)
    print("Attention-like Shape Test: [B, H, S, S]")
    print("=" * 60)

    torch.manual_seed(42)

    dtype = torch.float16
    device = "cuda"

    test_cases = [
        (1, 8, 512),
        (1, 16, 1024),
        (2, 16, 1024),
        (1, 16, 2048),
        (1, 32, 2048),
    ]

    for B, H, S in test_cases:
        print(f"\nConfiguration: B={B}, H={H}, S={S}, shape=[B,H,S,S], dtype={dtype}")

        x = torch.randn((B, H, S, S), device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(S)

        run_one_perf_case(
            x,
            mode_name="attn_scale_softmax",
            scale=scale,
        )

        run_one_perf_case(
            x,
            mode_name="attn_causal_softmax",
            scale=scale,
            is_causal=True,
        )

        # mask case 比较吃显存和时间，只在 S <= 1024 时测。
        if S <= 1024:
            bool_mask = make_bool_mask(x.shape, device=device)
            additive_mask = make_additive_mask(x.shape, device=device, dtype=dtype)

            run_one_perf_case(
                x,
                mode_name="attn_bool_mask_softmax",
                scale=scale,
                mask=bool_mask,
            )

            run_one_perf_case(
                x,
                mode_name="attn_additive_mask_softmax",
                scale=scale,
                mask=additive_mask,
            )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Step 1: Running Accuracy Test...")
    softmax_accuracy_test()

    print("\nStep 2: Running Performance Test...")
    test_softmax_performance()

    print("\nStep 3: Running 3D Shape Test...")
    test_softmax_3d_shape()

    print("\nStep 4: Running Attention-like Shape Test...")
    test_softmax_attention_like_shape()