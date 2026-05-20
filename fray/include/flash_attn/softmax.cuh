#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include "utils.cuh"

namespace fray{
    
using namespace cute;
using namespace fray::utils;

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
        CUTE_UNROLL
        for(int r = 0; r < size<0>(scores);++r){
            for(int c = 0; c < size<1>(scores); ++c){
                cur_max(r) = fmaxf(cur_max(r), scores(r, c));
            }
        }

        CUTE_UNROLL
        for(int r = 0; r < kNumRows; ++r){
            cur_max(r) = warp_reduce<4>(cur_max(r), MaxOp{});
        }

        CUTE_UNROLL
        for(int r = 0; r < kNumRows; ++r){
            float prev_max = row_max(r);
            row_max(r) = fmaxf(prev_max, cur_max(r));

            float scale = (row_max(r) == -INFINITY) ? 0.0f : exp2f((prev_max - row_max(r)) * scale_log2);
            row_sum(r) *= scale;
            CUTE_UNROLL
            for(int c = 0;c < size<1>(output); ++c){
                output(r, c) *= scale;
            }
        }

        CUTE_UNROLL
        for(int r = 0; r < size<0>(scores); ++r){
            float m_log2 = row_max(r) * scale_log2;
            float l_sum = 0.0f;
            CUTE_UNROLL
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

        CUTE_UNROLL
        for(int r = 0; r < kNumRows; ++r){
            float inv_sum = (row_sum(r) > 0) ? (1.0f / row_sum(r)) : 1.0f;
            CUTE_UNROLL
            for(int c = 0; c < size<1>(output); ++c){
                output(r, c) *= inv_sum;
            }
        }
    }
};

} // namespace fray