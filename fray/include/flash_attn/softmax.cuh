#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

namespace fray{
using namespace cute;

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

template <int N, typename T, typename ReduceOp>
__device__ T warp_reduce(T val, ReduceOp op){
    #pragma unroll
    for(int offset = N / 2;offset > 0; offset >>=1){
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Convert from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
__device__ auto mma_to_rowcol_layout(Layout acc_layout){
    auto l = logical_divide(acc_layout, Shape<_2>{}); // l: ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), // 逻辑行
                       make_layout(get<0, 0>(l), get<2>(l)));// 逻辑列
}

template <int kNumRows>
struct OnlineSoftmax{
    using StatTensor = decltype(make_tensor<float>(Shape<Int<kNumRows>>{}));
    StatTensor row_max, row_sum;
    
    __device__ OnlineSoftmax() {
        cute::fill(row_max, -INFINITY);
        cute::fill(row_sum, 0.0f);
    }

    template <typename TensorS, typename TensorO>
    __device__ void update(TensorS& accum_s, TensorO& accum_o, float scale_log2){
        auto scores = make_tensor(accum_s.data(), mma_to_rowcol_layout(accum_s.layout()));
        auto output = make_tensor(accum_o.data(), mma_to_rowcol_layout(accum_o.layout()));

        StatTensor cur_max;
        fill(cur_max, -INFINITY);
        for(int r = 0; r < size<0>(scores);++r){
            for(int c = 0; c < size<1>(scores); ++c){
                cur_max(r) = fmaxf(cur_max(r), scores(r, c));
            }
        }

        for(int r = 0; r < kNumRows; ++r){
            cur_max(r) = warp_reduce<4>(cur_max(r), MaxOp{});
        }

        for(int r = 0; r < kNumRows; ++r){
            float prev_max = row_max(r);
            row_max(r) = fmaxf(prev_max, cur_max(r));

            float scale = (row_max(r) == -INFINITY) ? 0.0f : exp2f((prev_max - row_max(r)) * scale_log2);
            row_sum(r) *= scale;
            for(int c = 0;c < size<1>(output); ++c){
                output(r, c) *= scale;
            }
        }

        for(int r = 0; r < size<0>(scores); ++r){
            float m_log2 = row_max(r) * scale_log2;
            float l_sum = 0.0f;
            for(int c = 0; c < size<1>(scores); ++c){
                scores(r, c) = exp2f(scores(r, c) * scale_log2 - m_log2);
                l_sum += scores(r, c);
            }

            row_sum(r) += warp_reduce<4>(l_sum, SumOp{});
        }
    }

    template <typename TensorO>
    __device__ void finalize(TensorO& accum_o){
        auto output = make_tensor(accum_o.data(), mma_to_rowcol_layout(accum_o.layout()));

        for(int r = 0; r < kNumRows; ++r){
            float inv_sum = (row_sum(r) > 0) ? (1.0f / row_sum(r)) : 1.0f;
            for(int c = 0; c < size<1>(output); ++c){
                output(r, c) *= inv_sum;
            }
        }
    }
};

} // namespace fray