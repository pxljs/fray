#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#include "reduce_utils.cuh"

namespace fray{

using namespace fray::utils;

union alignas(16) Float4Pack {
    uint4 u4;
    half2 h2[4];
};

template <int VECS_PER_THREAD>
__global__ void fused_rmsnorm_kernel(
    half* __restrict__ output, 
    half* __restrict__ residual,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    float epsilon,
    int D)
{
    // 每个block处理一个token（矩阵的一行）
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const uint4* in_ptr = reinterpret_cast<const uint4*>(input + row * D);
    uint4* res_ptr = reinterpret_cast<uint4*>(residual + row * D);
    uint4* out_ptr = reinterpret_cast<uint4*>(output + row * D);
    const uint4* w_ptr = reinterpret_cast<const uint4*>(weight);

    // 物理寄存器缓存
    float local_cache[VECS_PER_THREAD][8];

    float sum_sq = 0.0f;

    #pragma unroll
    for(int i = 0; i < VECS_PER_THREAD; i++){
        int col_vec_idx = i * blockDim.x + tid;

        if(col_vec_idx < D / 8){
            Float4Pack in_pack, res_pack, res_new_pack;

            in_pack.u4 = in_ptr[col_vec_idx];
            res_pack.u4 = res_ptr[col_vec_idx];

            #pragma unroll
            for(int k = 0; k < 4; k++){
                float2 f_in  = __half22float2(in_pack.h2[k]);
                float2 f_res = __half22float2(res_pack.h2[k]);

                // x_new = x_old + residual
                float2 f_new;
                f_new.x = f_in.x + f_res.x;
                f_new.y = f_in.y + f_res.y;

                local_cache[i][k * 2 + 0] = f_new.x;
                local_cache[i][k * 2 + 1] = f_new.y;

                // 累加平方和
                sum_sq += f_new.x * f_new.x + f_new.y * f_new.y;

                // 重新打包准备写回
                res_new_pack.h2[k] = __floats2half2_rn(f_new.x, f_new.y);
            }
            res_ptr[col_vec_idx] = res_new_pack.u4;
        }
    }

    sum_sq = block_reduce<float>(sum_sq, SumOp<float>{});

    __shared__ float s_rms;
    if(tid == 0){
        // RMS = 1 / sqrt(Variance + epsilon)
        s_rms = rsqrtf(sum_sq / (float)D + epsilon);
    }
    __syncthreads();

    float rms = s_rms;

    #pragma unroll
    for(int i = 0; i < VECS_PER_THREAD; ++i){
        int col_vec_idx = i * blockDim.x + tid;
        
        if (col_vec_idx < D / 8) {
            Float4Pack w_pack, out_pack;
            
            w_pack.u4 = w_ptr[col_vec_idx];

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f_w = __half22float2(w_pack.h2[k]);

                // y = x_new * rms * gamma
                float2 f_out;
                f_out.x = local_cache[i][k * 2 + 0] * rms * f_w.x;
                f_out.y = local_cache[i][k * 2 + 1] * rms * f_w.y;

                out_pack.h2[k] = __floats2half2_rn(f_out.x, f_out.y);
            }

            out_ptr[col_vec_idx] = out_pack.u4;
        }
    }
}

void fused_rmsnorm(half* output, half* residual, const half* input, const half* weight, 
                   float epsilon, int num_tokens, int hidden_dim, cudaStream_t stream = 0) 
{
    // 每个 uint4 包含 8 个 half
    const int VEC_SIZE = 8;
    
    if (hidden_dim % VEC_SIZE != 0) {
        std::cerr << "Error: hidden_dim must be a multiple of 8 for vectorization." << std::endl;
        return;
    }

    int num_vecs = hidden_dim / VEC_SIZE;
    
    // 一个 Block 处理一个 Token，最大线程数不超 1024
    int threads = num_vecs;
    int vecs_per_thread = 1;

    if (threads > 1024) {
        vecs_per_thread = (threads + 1023) / 1024;
        threads = (threads + vecs_per_thread - 1) / vecs_per_thread;
    }

    dim3 grid(num_tokens);
    dim3 block(threads);

    // 根据推导出的 vecs_per_thread 启动对应的静态模板 Kernel
    if (vecs_per_thread == 1) {
        fused_rmsnorm_kernel<1><<<grid, block, 0, stream>>>(output, residual, input, weight, epsilon, hidden_dim);
    } else if (vecs_per_thread == 2) {
        fused_rmsnorm_kernel<2><<<grid, block, 0, stream>>>(output, residual, input, weight, epsilon, hidden_dim);
    } else if (vecs_per_thread == 4) {
        fused_rmsnorm_kernel<4><<<grid, block, 0, stream>>>(output, residual, input, weight, epsilon, hidden_dim);
    } else if (vecs_per_thread == 8) {
        fused_rmsnorm_kernel<8><<<grid, block, 0, stream>>>(output, residual, input, weight, epsilon, hidden_dim);
    } else {
        std::cerr << "Error: hidden_dim is too large! Max supported is 65536." << std::endl;
    }

}

} // namespace fray