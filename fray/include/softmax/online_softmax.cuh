#pragma once

#include <cfloat>

namespace fray {

// 存储 Max 和 Sum 的结构体，对齐以优化寄存器传递
struct alignas(8) MD {
    float m;
    float d;
};

// 合并局部 MD
__device__ __forceinline__ MD merge_md(MD a, MD b) {
    float max_val = fmaxf(a.m, b.m);
    float sum_val = a.d * __expf(a.m - max_val) + b.d * __expf(b.m - max_val);
    return {max_val, sum_val};
}

__device__ __forceinline__ MD warp_reduce_md(MD val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        MD other;
        other.m = __shfl_down_sync(0xffffffff, val.m, offset);
        other.d = __shfl_down_sync(0xffffffff, val.d, offset);
        val = merge_md(val, other);
    }
    return val;
}

template <int BLOCK_SIZE, int T_VEC_SIZE>
__global__ void online_softmax_pro_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    const uint32_t D) 
{
    // 每个 Block 负责一行
    const int row_offset = blockIdx.x * D;
    const float4* row_in = reinterpret_cast<const float4*>(input + row_offset);
    float4* row_out = reinterpret_cast<float4*>(output + row_offset);

    // 寄存器暂存
    float4 frag[T_VEC_SIZE];
    MD local_md = {-FLT_MAX, 0.0f};

    // 读取 + Online 更新 Max/Sum
    #pragma unroll
    for (int i = 0; i < T_VEC_SIZE; ++i) {
        int vec_idx = threadIdx.x + i * BLOCK_SIZE;
        if (vec_idx * 4 < D) {
            float4 v = row_in[vec_idx];
            frag[i] = v; // 存入寄存器

            float m_v = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            float d_v = __expf(v.x - m_v) + __expf(v.y - m_v) + __expf(v.z - m_v) + __expf(v.w - m_v);
            
            local_md = merge_md(local_md, {m_v, d_v});
        }
    }

    // Block 级规约
    local_md = warp_reduce_md(local_md);

    __shared__ float shared_m[32];
    __shared__ float shared_d[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared_m[wid] = local_md.m;
        shared_d[wid] = local_md.d;
    }
    __syncthreads();

    MD final_md = {-FLT_MAX, 0.0f};
    if (wid == 0) {
        int num_warps = (BLOCK_SIZE + 31) / 32;
        final_md.m = (lane < num_warps) ? shared_m[lane] : -FLT_MAX;
        final_md.d = (lane < num_warps) ? shared_d[lane] : 0.0f;
        final_md = warp_reduce_md(final_md);
        if (lane == 0) {
            shared_m[0] = final_md.m;
            shared_d[0] = final_md.d;
        }
    }
    __syncthreads();

    float m_global = shared_m[0];
    float d_inv = 1.0f / shared_d[0];

    #pragma unroll
    for (int i = 0; i < T_VEC_SIZE; ++i) {
        int vec_idx = threadIdx.x + i * BLOCK_SIZE;
        if (vec_idx * 4 < D) {
            float4 v = frag[i];
            v.x = __expf(v.x - m_global) * d_inv;
            v.y = __expf(v.y - m_global) * d_inv;
            v.z = __expf(v.z - m_global) * d_inv;
            v.w = __expf(v.w - m_global) * d_inv;
            row_out[vec_idx] = v;
        }
    }
}

// Host 端调用
void online_softmax_c(float* input, float* output, int M, int D) {
    if (D <= 128) {
        online_softmax_pro_kernel<32, 1><<<M, 32>>>(input, output, D);
    } else if (D <= 512) {
        // 每个线程处理 1 个 float4, 128 线程正好处理 512 个元素
        online_softmax_pro_kernel<128, 1><<<M, 128>>>(input, output, D);
    } else if (D <= 1024) {
        // 256 线程，每个线程读 1 个 float4
        online_softmax_pro_kernel<256, 1><<<M, 256>>>(input, output, D);
    } else {
        // 对于超大 D (如 4096), 增加 T_VEC_SIZE
        online_softmax_pro_kernel<256, 4><<<M, 256>>>(input, output, D);
    }
}

} // namespace fray