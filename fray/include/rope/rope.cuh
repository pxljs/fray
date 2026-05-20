#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "reduce_utils.cuh"

namespace fray {

union alignas(16) Float4Pack {
    uint4 u4;
    half2 h2[4];
    half  h[8];
};


template <int HEAD_DIM>
__global__ void fused_rope_kernel(
    half* __restrict__ Q,           // [num_tokens, num_q_heads, head_dim]
    half* __restrict__ K,           // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ cos_table, // [max_seq_len, head_dim / 2]
    const float* __restrict__ sin_table, // [max_seq_len, head_dim / 2]
    const int* __restrict__ cache_offsets, // [num_tokens] 存储每个 token 在序列中的 pos
    int num_q_heads,
    int num_kv_heads,
    int stride_q, // num_heads * head_dim
    int stride_k  // num_kv_heads * head_dim
)
{
    // 每个 Block 处理一个 Token 的一个 Head 
    // blockIdx.y 对应 token_idx，blockIdx.x 对应 head_idx
    int token_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    const int half_dim = HEAD_DIM / 2;

    int pos = cache_offsets[token_idx];

    // 定位 Q 或 K 的起始位置
    // 有些 Block 可能负责 Q，有些可能负责 K
    bool is_query = head_idx < num_q_heads;
    half* data_ptr;
    int head_in_type_idx;

    if(is_query) {
        data_ptr = Q + token_idx * stride_q + head_idx * HEAD_DIM;
        head_in_type_idx = head_idx;
    }else{
        int kv_head_idx = head_idx - num_q_heads;
        if (kv_head_idx >= num_kv_heads) return; // 越界保护
        data_ptr = K + token_idx * stride_k + kv_head_idx * HEAD_DIM;
        head_in_type_idx = kv_head_idx;
    }

    // 向量化指针
    // 指向前半部分 [0 ... d/2-1]
    const uint4* first_half_ptr  = reinterpret_cast<const uint4*>(data_ptr);
    // 指向后半部分 [d/2 ... d-1]
    const uint4* second_half_ptr = reinterpret_cast<const uint4*>(data_ptr + half_dim);
    
    // 线程内循环处理（如果 HEAD_DIM 很大）
    #pragma unroll
    for (int i = tid * 8; i < half_dim; i += blockDim.x * 8) {
        Float4Pack v_first, v_second;
        
        // 同时发起两次 128-bit 读取
        v_first.u4  = first_half_ptr[i / 8];
        v_second.u4 = second_half_ptr[i / 8];

        // 读取对应的 sin/cos (通常在 FP32)
        // 表的形状是 [max_seq_len, d/2]
        const float* c_ptr = cos_table + pos * half_dim + i;
        const float* s_ptr = sin_table + pos * half_dim + i;

        Float4Pack res_first, res_second;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float x_i = __half2float(v_first.h[k]);
            float x_j = __half2float(v_second.h[k]);
            
            float cos_val = c_ptr[k];
            float sin_val = s_ptr[k];

            // 核心数学公式：旋转变换
            // x_i_new = x_i * cos - x_j * sin
            // x_j_new = x_i * sin + x_j * cos
            res_first.h[k]  = __float2half(x_i * cos_val - x_j * sin_val);
            res_second.h[k] = __float2half(x_i * sin_val + x_j * cos_val);
        }

        // 向量化写回
        uint4* first_half_out_ptr  = reinterpret_cast<uint4*>(data_ptr);
        uint4* second_half_out_ptr = reinterpret_cast<uint4*>(data_ptr + half_dim);
        
        first_half_out_ptr[i / 8]  = res_first.u4;
        second_half_out_ptr[i / 8] = res_second.u4;
    }
}

void fused_rope(half* Q, half* K, const float* cos_table, const float* sin_table,
                const int* cache_offsets, int num_tokens, int num_heads, 
                int num_kv_heads, int head_dim, cudaStream_t stream) 
{
    // 每个 Head 分配 1 个 Warp (32 线程) 是最平衡的
    // 对于 head_dim=128，每个线程实际只需处理 128/2/32 = 2 个对
    dim3 block(32);
    // Grid.y 负责 Token，Grid.x 负责所有的 Head (Q + K)
    dim3 grid(num_heads + num_kv_heads, num_tokens);

    // 根据 head_dim 派发模板（优化：可以处理任意 head_dim，这里以 128 为例）
    if (head_dim == 128) {
        fused_rope_kernel<128><<<grid, block, 0, stream>>>(
            Q, K, cos_table, sin_table, cache_offsets, 
            num_heads, num_kv_heads, num_heads * 128, num_kv_heads * 128);
    }
}

} // namespace fray