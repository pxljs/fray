#pragma once

#include <cute/tensor.hpp>
#include <cfloat>
#include "reduce/reduce.cuh"

using namespace cute;

namespace fray{

// 由于head_dim通常比较小（128、256、512...），因此一个block处理一行
template <typename T>
__global__ void softmax_2pass_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int B, int H, int S, int D,
    int stride_b, int stride_h, int stride_s, int stride_d)
{
    auto layout = make_layout(
        make_shape(make_shape(B, H, S), D),
        make_stride(make_stride(stride_b, stride_h, stride_s), stride_d)
    );

    auto tensor_in = make_tensor(make_gmem_ptr(input), layout);
    auto tensor_out = make_tensor(make_gmem_ptr(output), layout);

    int row_idx = blockIdx.x;
    if(row_idx >= B * H * S) return;


    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    __shared__ float s_max[32];
    __shared__ float s_sum[32];

    // Pass1: find row_max
    float thread_max = -FLT_MAX;

    for(int k = tid;k < D; k += blockDim.x){
        float val = static_cast<float>(tensor_in(row_idx, k));
        thread_max = fmaxf(thread_max, val);
    }

    thread_max = warp_reduce(thread_max, MaxOp<float>());
    if(lane == 0) s_max[wid] = thread_max;
    __syncthreads();

    float row_max = -FLT_MAX;
    if(wid == 0) {
        row_max = (tid < (blockDim.x / 32))?s_max[lane]:-FLT_MAX;
        row_max = warp_reduce(row_max, MaxOp<float>());
        if(lane == 0) s_max[0] = row_max;
    }
    __syncthreads();
    row_max = s_max[0];

    // Pass2: Cal sum and write to output
    float thread_sum = 0.0f;
    for(int k = tid;k < D;k+=blockDim.x){
        float val = static_cast<float>(tensor_in(row_idx, k));
        thread_sum += exp(val - row_max);
    }

    thread_sum = warp_reduce(thread_sum, SumOp<float>());

    if(lane == 0) s_sum[wid] = thread_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if(wid == 0){
        row_sum = (tid < (blockDim.x / 32)) ? s_sum[lane] : 0.0f;
        row_sum = warp_reduce(row_sum, SumOp<float>());
        if (lane == 0) s_sum[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_sum[0];

    for(int k = tid;k < D;k += blockDim.x){
        float val = static_cast<float>(tensor_in(row_idx,k));
        float softmax_val = expf(val - row_max) / row_sum;
        tensor_out(row_idx,k) = static_cast<T>(softmax_val);
    }
}

template <typename T, const int BLOCK_SIZE = 512>
void softmax_c(
    T *d_input, T *d_output, 
    int B, int H, int S, int D,
    int stride_b, int stride_h, int stride_s, int stride_d,
    cudaStream_t stream = 0)
{
    int total_rows = B * H * S;
    
    int blocks = total_rows;

    softmax_2pass_kernel<T><<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_output,
        B, H, S, D,
        stride_b, stride_h, stride_s, stride_d
    );
}
} // namespace fray