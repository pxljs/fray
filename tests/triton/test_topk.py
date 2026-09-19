import pytest
import torch

from fray.triton import topk, routed_grouped_gemm, moe_select_topk_softmax_with_counts

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("algorithm", ["auto", "torch"])
def test_cpu_fallback(algorithm):
    x = torch.randn(4, 17, requires_grad=True)
    result = topk(x, 4, algorithm=algorithm)
    torch.testing.assert_close(result.values, torch.topk(x, 4).values)
    result.values.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x).scatter_(-1, result.indices, 1))
    with pytest.raises(ValueError):
        topk(x, 4, algorithm="sort")
    with pytest.raises(ValueError):
        topk(x, 18)


def test_out_validation():
    x = torch.randn(3, 8)
    with pytest.raises(ValueError, match="overlap"):
        topk(x, 8, out=(x, torch.empty_like(x, dtype=torch.int64)))
    out = (torch.empty(3, 4), torch.empty(3, 4, dtype=torch.int64))
    assert topk(x, 4, out=out).values is out[0]
    assert topk(torch.empty(3, 0), 0).values.shape == (3, 0)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("n,k", [(31, 8), (129, 65), (1025, 1025), (4097, 2049), (16384, 12000)])
def test_topk_cuda(monkeypatch, dtype, largest, n, k):
    x = torch.randint(-20, 21, (3, n), device="cuda").to(dtype)
    x[:, :5] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0.], device=x.device)
    ids = torch.argsort(x, stable=True, descending=largest)[:, :k]
    values = x.gather(-1, ids)
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected torch fallback")
    monkeypatch.setattr(torch, "topk", forbidden)
    for algorithm in ["auto", "sort"] + (["reduce"] if k <= 32 else []):
        result = topk(x, k, largest, algorithm=algorithm)
        torch.testing.assert_close(result.indices, ids)
        torch.testing.assert_close(result.values, values, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(result.values), torch.signbit(values))


@CUDA
@pytest.mark.parametrize("e,k", [(17, 4), (17, 12), (17, 17), (4097, 64)])
def test_fused_routing(e, k):
    logits = torch.randn(7, e, device="cuda")
    expected = torch.topk(logits, k)
    counts = torch.full((e,), 999, device="cuda", dtype=torch.int64)
    for _ in range(2):
        ids, probs, actual_counts = moe_select_topk_softmax_with_counts(logits, k, counts=counts)
        torch.testing.assert_close(ids, expected.indices)
        torch.testing.assert_close(probs, expected.values.softmax(-1))
        torch.testing.assert_close(actual_counts, torch.bincount(expected.indices.flatten(), minlength=e))


@CUDA
@pytest.mark.parametrize("k", [2, 12])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("combine", ["auto", "atomic", "staged", "scatter"])
def test_routed_grouped_gemm(k, dtype, combine):
    x = torch.randn(13, 64, device="cuda", dtype=dtype) * .1
    weights = torch.randn(17, 64, 48, device="cuda", dtype=dtype) * .1
    logits = torch.randn(13, 17, device="cuda")
    actual = routed_grouped_gemm(x, weights, logits, k, combine=combine)
    # Independent PyTorch routing, expert computation and weighted combination.
    vals, ids = torch.topk(logits, k)
    probs = vals.softmax(-1)
    expected = torch.zeros(13, 48, device=x.device, dtype=dtype)
    for expert in range(17):
        tokens, routes = torch.where(ids == expert)
        products = x[tokens] @ weights[expert]
        expected.index_add_(0, tokens, (products.float() * probs[tokens, routes, None]).to(dtype))
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.005)


@CUDA
def test_stream_graph():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(5, 129, device="cuda")
        out = (torch.empty(5, 65, device="cuda"), torch.empty(5, 65, device="cuda", dtype=torch.int64))
        topk(x, 65, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            topk(x, 65, out=out)
        x.neg_()
        graph.replay()
        ref = torch.topk(x, 65)
    stream.synchronize()
    torch.testing.assert_close(out[0], ref.values, rtol=0, atol=0)
    torch.testing.assert_close(out[1], ref.indices)


@CUDA
@pytest.mark.parametrize("k", [2, 12])
def test_full_moe_from_logits(k):
    from fray.triton import fused_moe
    x = torch.randn(9, 64, device="cuda", dtype=torch.float16) * .1
    w13 = torch.randn(17, 64, 64, device="cuda", dtype=x.dtype) * .1
    w2 = torch.randn(17, 32, 64, device="cuda", dtype=x.dtype) * .1
    logits = torch.randn(9, 17, device="cuda")
    actual = fused_moe(x, None, None, w13, w2, router_logits=logits, top_k=k,
                       persistent_waves=1)
    values, ids = torch.topk(logits, k)
    probs = values.softmax(-1)
    expected = torch.zeros_like(x)
    for e in range(17):
        tokens, routes = torch.where(ids == e)
        projection = x[tokens] @ w13[e]
        hidden = (torch.nn.functional.silu(projection[:, :32].float()) * projection[:, 32:].float()).to(x.dtype)
        products = hidden @ w2[e]
        expected.index_add_(0, tokens, (products.float() * probs[tokens, routes, None]).to(x.dtype))
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.005)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,k", [(129, 16), (1025, 13), (4097, 33), (16384, 64)])
@pytest.mark.parametrize("largest", [True, False])
def test_partial(dtype, n, k, largest):
    x = torch.randint(-10, 11, (3, n), device="cuda").to(dtype)
    x[:, :5] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0.], device=x.device)
    ids = torch.argsort(x, stable=True, descending=largest)[:, :k]
    from fray.triton.topk import _standalone_algorithm
    assert _standalone_algorithm(n, k) == ("partial" if n > 1024 else "sort")
    result = topk(x, k, largest, algorithm="partial")
    torch.testing.assert_close(result.indices, ids)
    torch.testing.assert_close(result.values, x.gather(-1, ids), rtol=0, atol=0, equal_nan=True)


@CUDA
@pytest.mark.parametrize("algorithm,k", [("partial", 16), ("sort", 16), ("sort", 5000)])
def test_workspace_reuse(algorithm, k):
    import triton
    x = torch.randn(3, 8193, device="cuda", dtype=torch.float16)
    count = (triton.cdiv(8193, 256) * triton.next_power_of_2(k) if algorithm == "partial"
             else triton.cdiv(8193, 4096) * 4096)
    workspace = torch.empty(3 * count, device=x.device, dtype=torch.uint32)
    for _ in range(2):
        x.neg_()
        result = topk(x, k, algorithm=algorithm, workspace=workspace)
        torch.testing.assert_close(result.values, torch.topk(x, k).values, rtol=0, atol=0)
        torch.testing.assert_close(result.values, x.gather(-1, result.indices), rtol=0, atol=0)
    with pytest.raises(ValueError, match="workspace"):
        topk(x, k, algorithm=algorithm, workspace=workspace[:1])


@CUDA
@pytest.mark.parametrize("combine", ["atomic", "staged"])
def test_routed_graph_no_host_metadata(monkeypatch, combine):
    import importlib
    module = importlib.import_module("fray.triton.grouped_gemm")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU metadata builder used")
    monkeypatch.setattr(module, "build_grouped_gemm_metadata", forbidden)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(13, 64, device="cuda", dtype=torch.float16) * .1
        weights = torch.randn(17, 64, 48, device="cuda", dtype=x.dtype) * .1
        logits = torch.randn(13, 17, device="cuda")
        routed_grouped_gemm(x, weights, logits, 4, combine=combine)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = routed_grouped_gemm(x, weights, logits, 4, combine=combine)
        x.neg_()
        graph.replay()
        reference = routed_grouped_gemm(x, weights, logits, 4, combine=combine)
    stream.synchronize()
    torch.testing.assert_close(result, reference, rtol=.03, atol=.005)


def test_standalone_router():
    from fray.triton.topk import _standalone_algorithm
    assert _standalone_algorithm(128, 8) == "reduce"
    assert _standalone_algorithm(4096, 16) == "partial"
    assert _standalone_algorithm(16384, 12000) == "sort"


@CUDA
@pytest.mark.parametrize("combine", ["atomic", "staged"])
def test_routed_tails_and_skew(combine):
    x = torch.randn(97, 65, device="cuda", dtype=torch.float16) * .1
    weights = torch.randn(17, 65, 129, device="cuda", dtype=x.dtype) * .1
    # All tokens choose the same two experts; other experts have no tiles.
    logits = torch.arange(17., device="cuda").expand(97, -1).contiguous()
    result = routed_grouped_gemm(x, weights, logits, 2, combine=combine)
    selected, ids = torch.topk(logits, 2)
    probs = selected.softmax(-1)
    expected = torch.zeros(97, 129, device=x.device, dtype=torch.float32)
    for expert in range(17):
        tokens, routes = torch.where(ids == expert)
        products = x[tokens] @ weights[expert]
        contribution = (products.float() * probs[tokens, routes, None]).to(x.dtype)
        expected.index_add_(0, tokens, contribution.float())
    torch.testing.assert_close(result, expected.to(x.dtype), rtol=.03, atol=.005)


@CUDA
def test_dispatch_inverse_positions():
    from fray.triton import build_moe_dispatch_metadata_fast
    ids = torch.tensor([[0, 2, 1], [2, 1, 0], [1, 0, 2]], device="cuda")
    weights = torch.arange(9., device="cuda").reshape(3, 3)
    positions = torch.empty(9, device="cuda", dtype=torch.int64)
    for _ in range(2):
        tokens, probabilities, offsets, counts = build_moe_dispatch_metadata_fast(
            ids, weights, 3, route_positions=positions)
        torch.testing.assert_close(tokens[positions], torch.arange(3, device="cuda").repeat_interleave(3))
        torch.testing.assert_close(probabilities[positions], weights.flatten())
        torch.testing.assert_close(positions.sort().values, torch.arange(9, device="cuda"))
        torch.testing.assert_close(offsets, torch.tensor([0, 3, 6, 9], device="cuda"))
        torch.testing.assert_close(counts, torch.tensor([3, 3, 3], device="cuda"))
    with pytest.raises(ValueError, match="storage"):
        build_moe_dispatch_metadata_fast(ids, weights, 3, route_positions=ids.view(-1))


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gather_combine_fixed_order(dtype):
    from fray.triton.grouped_gemm import _gather_combine_kernel
    positions = torch.randperm(7 * 13, device="cuda").reshape(7, 13)
    routed = torch.randn(7 * 13, 133, device="cuda", dtype=dtype)
    result = torch.full((7, 133), float('nan'), device="cuda", dtype=dtype)
    _gather_combine_kernel[(7, 2)](routed, positions, result, 133, 13, 128, num_warps=4)
    expected = torch.zeros(7, 133, device="cuda")
    for k in range(13):
        expected += routed[positions[:, k]].float()
    torch.testing.assert_close(result, expected.to(dtype), rtol=0, atol=0)
