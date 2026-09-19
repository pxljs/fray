"""PYTHONPATH=. .venv/bin/python tests/triton/bench_topk.py [--grouped]"""
import argparse

import torch
import triton

from fray.triton import topk, routed_grouped_gemm
from fray.triton.topk import _standalone_algorithm


def torch_routed(x, weights, logits, k):
    values, ids = torch.topk(logits, k)
    probabilities = values.softmax(-1)
    out = torch.zeros(x.shape[0], weights.shape[2], device=x.device, dtype=x.dtype)
    for expert in range(weights.shape[0]):
        tokens, routes = torch.where(ids == expert)
        products = x[tokens] @ weights[expert]
        out.index_add_(0, tokens, (products.float() * probabilities[tokens, routes, None]).to(x.dtype))
    return out


def breakdown(x, weights, logits, k, combine):
    from fray.triton.fused_moe import (
        moe_select_topk_softmax_with_counts, build_moe_dispatch_metadata_fast,
        build_grouped_tile_offsets_no_sync,
    )
    from fray.triton.grouped_gemm import _routed_gemm_kernel, _gather_combine_kernel
    ids, probs, counts = moe_select_topk_softmax_with_counts(logits, k)
    atomic = k <= 4 if combine == "auto" else combine == "atomic"
    gather = not atomic and combine != "scatter"
    positions = torch.empty(x.shape[0]*k, device=x.device, dtype=torch.int64) if gather else None
    tokens, probabilities, offsets, _ = build_moe_dispatch_metadata_fast(
        ids, probs, weights.shape[0], counts=counts, route_positions=positions)
    n = weights.shape[2]
    tiles, programs = build_grouped_tile_offsets_no_sync(offsets, n, 32, 64, tokens.numel(), 3)
    atomic = k <= 4 if combine == "auto" else combine == "atomic"
    output = (torch.empty(x.shape[0], n, device=x.device, dtype=x.dtype) if gather else
              torch.zeros(x.shape[0], n, device=x.device, dtype=torch.float32))
    routed = output if atomic else torch.empty(tokens.numel(), n, device=x.device, dtype=x.dtype)
    def routing():
        return moe_select_topk_softmax_with_counts(logits, k)
    def dispatch():
        temporary_positions = torch.empty_like(positions) if gather else None
        return build_moe_dispatch_metadata_fast(ids, probs, weights.shape[0], counts=counts,
                                               route_positions=temporary_positions)
    def tile_metadata():
        return build_grouped_tile_offsets_no_sync(offsets, n, 32, 64, tokens.numel(), 3)
    def gemm():
        if atomic:
            output.zero_()  # reset on every benchmark iteration
        _routed_gemm_kernel[(programs,)](x, weights, routed, tokens, probabilities, offsets, tiles,
                                        x.shape[1], n, weights.shape[0], 32, 64, 32, atomic, num_warps=4)
    def finish():
        if gather:
            _gather_combine_kernel[(x.shape[0], triton.cdiv(n, 128))](
                routed, positions, output, n, k, 128, num_warps=4)
        elif not atomic:
            output.zero_()
            output.index_add_(0, tokens, routed.float())
        return output.to(x.dtype)
    gemm()
    for name, fn in [("routing+counts", routing), ("dispatch", dispatch),
                     ("tile metadata", tile_metadata), ("GEMM+weights(+atomic/reset)", gemm),
                     ("combine/cast", finish)]:
        fn()
        print("stage", name, triton.testing.do_bench_cudagraph(fn)*1000, "us warm graph")
    print("Stage times use prepared inputs; their sum is not eager end-to-end latency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--breakdown', action='store_true')
    parser.add_argument('--combine', choices=['auto', 'atomic', 'staged', 'scatter'], default='auto')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(1, 'CUDA unavailable; run on the target GPU.\n')
    print(torch.cuda.get_device_name(), 'Triton', triton.__version__)
    if args.grouped or args.breakdown:
        # Full eager wall time: include routing, GPU metadata, allocation,
        # GEMM and combination for both implementations; exclude warm-up.
        import time
        x = torch.randn(128, 128, device='cuda', dtype=torch.float16) * .1
        weights = torch.randn(17, 128, 128, device='cuda', dtype=x.dtype) * .1
        logits = torch.randn(128, 17, device='cuda')
        for k in (2, 12):
            def custom():
                return routed_grouped_gemm(x, weights, logits, k, combine=args.combine)
            def reference():
                return torch_routed(x, weights, logits, k)
            torch.testing.assert_close(custom(), reference(), rtol=.03, atol=.005)
            if args.breakdown:
                breakdown(x, weights, logits, k, args.combine)
            for name, fn in [('triton', custom), ('torch', reference)]:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                print('routed grouped GEMM', k, name, (time.perf_counter()-start)*1e6/20, 'us eager wall')
    else:
        for n, k in [(128, 8), (1024, 16), (4096, 16), (16384, 64), (4096, 512), (16384, 12000)]:
            x = torch.randn(128, n, device='cuda', dtype=torch.float16)
            out = (torch.empty(128, k, device=x.device, dtype=x.dtype),
                   torch.empty(128, k, device=x.device, dtype=torch.int64))
            ref = tuple(torch.empty_like(t) for t in out)
            torch.topk(x, k, out=ref)
            for algorithm in ['auto', 'sort', 'torch'] + (['reduce'] if k <= 32 else []) + (['partial'] if k <= 64 else []):
                selected = _standalone_algorithm(n, k) if algorithm == 'auto' else algorithm
                scratch = selected == 'partial' or (selected == 'sort' and n > 4096)
                count = triton.cdiv(n, 256)*triton.next_power_of_2(k) if selected == 'partial' else triton.cdiv(n, 4096)*4096
                workspace = torch.empty(128*count, device=x.device, dtype=torch.uint32) if scratch else None
                def run():
                    return topk(x, k, algorithm=algorithm, out=out, workspace=workspace)
                result = run()
                torch.testing.assert_close(result.values, ref[0], rtol=0, atol=0)
                torch.testing.assert_close(result.values, x.gather(-1, result.indices), rtol=0, atol=0)
                us = triton.testing.do_bench_cudagraph(run) * 1000
                print(n, k, algorithm, us, 'us warm graph, preallocated')


if __name__ == '__main__':
    main()
