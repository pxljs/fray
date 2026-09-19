import pytest
import torch
from fray.jit_kernels.topk import topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("n,k", [(1, 1), (31, 8), (129, 32), (1024, 16),
                                (1025, 8), (4096, 32), (16384, 8),
                                (64, 0), (64, 64), (16385, 4)])
def test_topk(dtype, largest, n, k):
    x = torch.randn(7, n, device="cuda", dtype=dtype)
    values, indices = topk(x, k, largest)
    ref = torch.topk(x, k, largest=largest)
    torch.testing.assert_close(values, ref.values, rtol=0, atol=0)
    torch.testing.assert_close(values, x.gather(-1, indices), rtol=0, atol=0)
    if k > 1:
        assert (indices.sort().values.diff() > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("n", [33, 1025])
def test_special_values_and_stream(largest, n):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.zeros(5, n, device="cuda")
        x[:, :6] = torch.tensor([float('nan'), float('inf'), -float('inf'),
                                -0., 0., float('nan')], device="cuda")
        expected = torch.argsort(x, descending=largest, stable=True)[:, :32]
        out = (torch.empty(5, 32, device="cuda"),
               torch.empty(5, 32, device="cuda", dtype=torch.int64))
        values, indices = topk(x, 32, largest, out=out)
        torch.testing.assert_close(indices, expected)
        torch.testing.assert_close(values, x.gather(-1, indices), equal_nan=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            topk(x, 32, largest, out=out)
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(indices, expected)


def test_cpu_fallback_and_validation():
    x = torch.randn(3, 9).t()
    a, b = topk(x, 2)
    ref = torch.topk(x, 2)
    torch.testing.assert_close(a, ref.values)
    torch.testing.assert_close(b, ref.indices)
    with pytest.raises(ValueError):
        topk(x, -1)
    with pytest.raises(ValueError):
        topk(x, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_and_alias():
    for shape in [(0, 128), (3, 0)]:
        x = torch.empty(shape, device="cuda")
        assert topk(x, 0)[0].shape == (shape[0], 0)
    x = torch.randn(3, 16, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        topk(x, 16, out=(x, torch.empty_like(x, dtype=torch.int64)))
    x = torch.randn(2, 7, 32, device="cuda").transpose(0, 1)
    torch.testing.assert_close(topk(x, 4)[0], torch.topk(x, 4).values)


@pytest.mark.parametrize("kwargs,error", [
    ({"k": True}, TypeError), ({"k": 1.5}, TypeError),
    ({"k": -1}, ValueError), ({"k": 10}, ValueError),
    ({"dim": True}, TypeError), ({"dim": 0.5}, TypeError),
    ({"dim": 2}, IndexError), ({"dim": -3}, IndexError),
    ({"largest": 1}, TypeError), ({"sorted": "yes"}, TypeError),
])
def test_argument_validation(kwargs, error):
    options = {"k": 2, **kwargs}
    with pytest.raises(error):
        topk(torch.randn(3, 9), **options)


def test_input_validation():
    with pytest.raises(TypeError, match="Tensor"):
        topk([1, 2, 3], 1)
    with pytest.raises(ValueError, match="dimension"):
        topk(torch.tensor(1.), 1)
    with pytest.raises(ValueError, match="strided"):
        topk(torch.eye(3).to_sparse(), 1)


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("sorted", [True, False])
def test_dim_and_named_result(dim, sorted):
    x = torch.arange(35.).reshape(5, 7)
    result = topk(x, 3, False, sorted, dim=dim)
    ref = torch.topk(x, 3, dim=dim, largest=False, sorted=sorted)
    torch.testing.assert_close(result.values, ref.values)
    torch.testing.assert_close(result.indices, ref.indices)


@pytest.mark.parametrize("case,error", [
    ("arity", TypeError), ("non_tensor", TypeError), ("shape", ValueError),
    ("value_dtype", ValueError), ("index_dtype", ValueError),
    ("non_contiguous", ValueError), ("input_alias", ValueError),
    ("output_alias", ValueError),
])
def test_out_validation(case, error):
    x = torch.randn(3, 8)
    values = torch.empty(3, 4)
    indices = torch.empty(3, 4, dtype=torch.int64)
    out = (values, indices)
    if case == "arity":
        out = (values,)
    elif case == "non_tensor":
        out = (None, indices)
    elif case == "shape":
        out = (torch.empty(3, 5), indices)
    elif case == "value_dtype":
        out = (values.double(), indices)
    elif case == "index_dtype":
        out = (values, indices.int())
    elif case == "non_contiguous":
        out = (torch.empty(4, 3).t(), indices)
    elif case == "input_alias":
        out = (x.view(-1)[:12].view(3, 4), indices)
    elif case == "output_alias":
        out = (indices.view(torch.float32).view(-1)[:12].view(3, 4), indices)
    with pytest.raises(error):
        topk(x, 4, out=out)


def test_out_identity_disjoint_storage_and_input_unchanged():
    # Disjoint contiguous slices in one storage are legal.
    storage = torch.randn(60)
    x = storage[:24].view(3, 8)
    original = x.clone()
    values = storage[24:36].view(3, 4)
    indices = torch.empty(3, 4, dtype=torch.int64)
    result = topk(x, 4, out=(values, indices))
    assert result.values is values and result.indices is indices
    torch.testing.assert_close(result.values, torch.topk(original, 4).values)
    torch.testing.assert_close(x, original)


@pytest.mark.parametrize("shape,k", [((0, 8), 4), ((2, 0), 0), ((8,), 0)])
def test_empty_cpu(shape, k):
    x = torch.empty(shape)
    result = topk(x, k)
    ref = torch.topk(x, k)
    torch.testing.assert_close(result.values, ref.values)
    torch.testing.assert_close(result.indices, ref.indices)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_autograd(device):
    x = torch.arange(24., device=device).reshape(3, 8).requires_grad_()
    result = topk(x, 3)
    result.values.sum().backward()
    expected = torch.zeros_like(x).scatter_(-1, result.indices, 1.)
    torch.testing.assert_close(x.grad, expected)
    out = (torch.empty(3, 3, device=device), torch.empty(3, 3, device=device, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="differentiation"):
        topk(x, 3, out=out)
    with torch.no_grad():
        actual = topk(x, 3, out=out)
        torch.testing.assert_close(actual.values, result.values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape,k", [((33,), 7), ((2, 3, 129), 16), ((9, 1025), 32)])
def test_custom_dispatch_out_and_reuse(monkeypatch, shape, k):
    x = torch.randn(*shape, device="cuda")
    original = x.clone()
    expected = torch.topk(x, k).values
    output_shape = (*shape[:-1], k)
    out = (torch.empty(output_shape, device=x.device),
           torch.empty(output_shape, device=x.device, dtype=torch.int64))
    def forbidden(*args, **kwargs):
        raise AssertionError("eligible inference input unexpectedly used torch.topk")
    monkeypatch.setattr(torch, "topk", forbidden)
    for _ in range(2):
        out[0].fill_(float('nan'))
        out[1].fill_(-1)
        result = topk(x, k, sorted=False, out=out)
        assert result.values is out[0] and result.indices is out[1]
        torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
        torch.testing.assert_close(result.values, x.gather(-1, result.indices), rtol=0, atol=0)
    torch.testing.assert_close(x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_updates_outputs():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(129., device="cuda").repeat(5, 1)
        out = (torch.empty(5, 8, device="cuda"),
               torch.empty(5, 8, device="cuda", dtype=torch.int64))
        topk(x, 8, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            topk(x, 8, out=out)
        x.neg_()
        graph.replay()
        expected = torch.topk(x, 8)
    stream.synchronize()
    torch.testing.assert_close(out[0], expected.values)
    torch.testing.assert_close(out[1], expected.indices)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_device_guard():
    with torch.cuda.device(0):
        x = torch.randn(5, 129, device="cuda:1")
        actual = topk(x, 8)
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual.values, torch.topk(x, 8).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fill", [0., float('inf'), -float('inf'), float('nan')])
@pytest.mark.parametrize("largest", [True, False])
def test_uniform_values(dtype, fill, largest):
    x = torch.full((5, 1057), fill, dtype=dtype, device="cuda")
    result = topk(x, 32, largest)
    expected_indices = torch.arange(32, device="cuda").expand(5, -1)
    torch.testing.assert_close(result.indices, expected_indices)
    torch.testing.assert_close(result.values, x[:, :32], rtol=0, atol=0, equal_nan=True)


def test_out_device_and_grad_validation():
    x = torch.randn(3, 8)
    with pytest.raises(ValueError, match="device"):
        topk(x, 4, out=(torch.empty(3, 4, device="meta"),
                       torch.empty(3, 4, dtype=torch.int64)))
    with pytest.raises(RuntimeError, match="differentiation"):
        topk(x, 4, out=(torch.empty(3, 4, requires_grad=True),
                       torch.empty(3, 4, dtype=torch.int64)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_launch_error_propagation(monkeypatch):
    import importlib
    module = importlib.import_module("fray.jit_kernels.topk")
    monkeypatch.setattr(module.jit_tuner, "compile_and_tune", lambda **kwargs: lambda *args: 9)
    with pytest.raises(RuntimeError, match="error 9"):
        topk(torch.randn(3, 128, device="cuda"), 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", ["width", "layout", "dtype", "dim", "grad"])
def test_fallback_dispatch(monkeypatch, case):
    x = torch.randn(5, 64, device="cuda")
    kwargs = dict(k=4)
    if case == "width":
        x = torch.randn(5, 16385, device="cuda")
    elif case == "k":
        kwargs["k"] = 33
    elif case == "layout":
        x = x.t()
    elif case == "dtype":
        x = x.double()
    elif case == "dim":
        kwargs["dim"] = 0
    else:
        x.requires_grad_()
    original = torch.topk
    calls = []
    def tracked(*args, **options):
        calls.append(options)
        return original(*args, **options)
    monkeypatch.setattr(torch, "topk", tracked)
    result = topk(x, **kwargs)
    assert len(calls) == 1
    ref = original(x, **kwargs)
    torch.testing.assert_close(result.values, ref.values)
    torch.testing.assert_close(result.indices, ref.indices)


@pytest.mark.parametrize("n,k,expected", [(128, 8, "reduce"), (128, 9, "radix"),
    (4096, 4, "reduce"), (4096, 5, "radix"), (16384, 8, "reduce"),
    (16384, 9, "radix"), (16, 16, "radix"), (16385, 1, "torch")])
def test_router(n, k, expected):
    from fray.jit_kernels.topk import _select_algorithm
    assert _select_algorithm(n, k) == expected


@pytest.mark.parametrize("algorithm", ["radix", "reduce"])
def test_forced_custom_rejects_cpu(algorithm):
    with pytest.raises(ValueError, match="forced custom"):
        topk(torch.randn(2, 128), 8, algorithm=algorithm)


def test_algorithm_validation():
    with pytest.raises(ValueError, match="algorithm"):
        topk(torch.randn(3, 8), 4, algorithm="invalid")
    with pytest.raises(ValueError, match="workspace"):
        topk(torch.randn(3, 8), 4, workspace=torch.empty(64, dtype=torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("n,k", [(33, 33), (129, 65), (1024, 513), (4096, 4096),
                                (4097, 65), (8193, 4097), (8193, 8193), (16384, 16000)])
def test_large_k_radix(monkeypatch, dtype, largest, n, k):
    # Discrete values force ties within and across tile boundaries.
    x = torch.randint(-20, 21, (3, n), device="cuda").to(dtype)
    x[:, :5] = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0.],
                            device=x.device, dtype=dtype)
    expected_ids = torch.argsort(x, descending=largest, stable=True)[:, :k]
    expected = x.gather(-1, expected_ids)
    def forbidden(*args, **kwargs):
        raise AssertionError("large-k unexpectedly used torch.topk")
    monkeypatch.setattr(torch, "topk", forbidden)
    for algorithm in ("auto", "radix"):
        result = topk(x, k, largest, algorithm=algorithm)
        torch.testing.assert_close(result.indices, expected_ids)
        torch.testing.assert_close(result.values, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(result.values), torch.signbit(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [128, 4096, 8193])
def test_reduce_radix_agreement(n):
    x = torch.randn(5, n, device="cuda")
    a = topk(x, 16, algorithm="reduce")
    b = topk(x, 16, algorithm="radix")
    torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
    torch.testing.assert_close(a.indices, b.indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tiled_workspace_graph():
    x = torch.randn(3, 8193, device="cuda")
    workspace = torch.empty(3 * 3 * 4096 * 8, device="cuda", dtype=torch.uint8)
    out = (torch.empty(3, 4097, device="cuda"),
           torch.empty(3, 4097, device="cuda", dtype=torch.int64))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        topk(x, 4097, out=out, workspace=workspace)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            topk(x, 4097, out=out, workspace=workspace)
        x.neg_()
        graph.replay()
        expected = torch.topk(x, 4097)
    stream.synchronize()
    torch.testing.assert_close(out[0], expected.values, rtol=0, atol=0)
    torch.testing.assert_close(out[0], x.gather(-1, out[1]), rtol=0, atol=0)
    with pytest.raises(ValueError, match="workspace"):
        topk(x, 4097, workspace=workspace[:1])
    with pytest.raises(ValueError, match="workspace"):
        topk(x, 4097, workspace=workspace[1:])
    with pytest.raises(ValueError, match="reduce"):
        topk(x, 33, algorithm="reduce")


def test_empty_nonlast_dimension_fallback():
    x = torch.empty(64, 0)
    result = topk(x, 64, dim=0)
    assert result.values.shape == (64, 0)
