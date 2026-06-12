#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "utils.cuh"
#include "reduce_utils.cuh"

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

    template <typename TensorS, typename... TensorOs>
    __device__ void update(TensorS& accum_s, float scale_log2, TensorOs&... accum_os) {
        auto layout_rc_s = mma_to_rowcol_layout(accum_s.layout());

        auto scale_outputs = [&](int r, float scale) {
            auto scale_single = [&](auto& accum_o) {
                auto layout_rc_o = mma_to_rowcol_layout(accum_o.layout());
                CUTE_UNROLL
                for (int c = 0; c < size<1>(layout_rc_o); ++c) {
                    accum_o(layout_rc_o(r, c)) *= scale;
                }
            };
            // C++17 折叠表达式：同时缩放传入的所有输出分块
            (scale_single(accum_os), ...);
        };

        StatTensor cur_max;
        fill(cur_max, -INFINITY);
        CUTE_UNROLL
        for (int r = 0; r < size<0>(layout_rc_s); ++r) {
            for (int c = 0; c < size<1>(layout_rc_s); ++c) {
                cur_max(r) = fmaxf(cur_max(r), accum_s(layout_rc_s(r, c)));
            }
        }

        // Warp 内部跨列规约得到行最大值
        CUTE_UNROLL
        for (int r = 0; r < kNumRows; ++r) {
            cur_max(r) = warp_reduce<4>(cur_max(r), MaxOp<float>{});
        }

        // 更新最大值并对所有的 O 寄存器分块应用统一的缩放因子
        CUTE_UNROLL
        for (int r = 0; r < kNumRows; ++r) {
            float prev_max = row_max(r);
            row_max(r) = fmaxf(prev_max, cur_max(r));

            float scale = (row_max(r) == -INFINITY) ? 0.0f : exp2f((prev_max - row_max(r)) * scale_log2);
            row_sum(r) *= scale;

            // 调用变参缩放
            scale_outputs(r, scale);
        }

        // 计算当前分块的注意力概率矩阵 P
        CUTE_UNROLL
        for (int r = 0; r < size<0>(layout_rc_s); ++r) {
            float m_log2 = row_max(r) * scale_log2;
            float l_sum = 0.0f;
            CUTE_UNROLL
            for (int c = 0; c < size<1>(layout_rc_s); ++c) {
                accum_s(layout_rc_s(r, c)) = exp2f(accum_s(layout_rc_s(r, c)) * scale_log2 - m_log2);
                l_sum += accum_s(layout_rc_s(r, c));
            }

            row_sum(r) += warp_reduce<4>(l_sum, SumOp<float>{});
        }
    }

    // 变参 finalize：支持同时对所有 O 分块执行除以最终归一化和的归一化
    template <typename... TensorOs>
    __device__ void finalize(TensorOs&... accum_os) {
        auto finalize_single = [&](auto& accum_o) {
            auto layout_rc = mma_to_rowcol_layout(accum_o.layout());
            CUTE_UNROLL
            for (int r = 0; r < kNumRows; ++r) {
                float inv_sum = (row_sum(r) > 0.0f) ? (1.0f / row_sum(r)) : 1.0f;
                CUTE_UNROLL
                for (int c = 0; c < size<1>(layout_rc); ++c) {
                    accum_o(layout_rc(r, c)) *= inv_sum;
                }
            }
        };
        (finalize_single(accum_os), ...);
    }
};

} // namespace fray