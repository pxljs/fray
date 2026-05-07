#pragma once

#include <cuda_fp16.h>
#include <cassert>
#include <cfloat>

namespace fray{

static constexpr uint32_t FULL_MASK = 0xffffffff;

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
    __device__ __forceinline__ T identity() const { return T(0); }
    __device__ __forceinline__ void atomic_op(T* address, T val) const { atomicAdd(address, val); }
};

template <>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half& a, const half& b) const { return __hadd(a, b); }
    // half2 SIMD add
    __device__ __forceinline__ half2 op2(const half2& a, const half2& b) const { return __hadd2(a, b); }
    // half2 reduce to half
    __device__ __forceinline__ half fold(const half2& a) const { return __hadd(a.x, a.y); }
    
    __device__ __forceinline__ half identity() const { return __float2half(0.0f); }

    __device__ __forceinline__ half2 identity2() const { return __float2half2_rn(0.0f); }

    __device__ __forceinline__ void atomic_op(half* address, half val) const {
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
        atomicAdd(address, val); // half atomicAdd
#else
        // fallback to naive
        unsigned short* address_as_ushort = (unsigned short*)address;
        unsigned short old = *address_as_ushort, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(__hadd(val, __ushort_as_half(assumed))));
        } while (assumed != old);
#endif
    }
};

template <typename T>
struct MaxOp;

template <>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const { return fmaxf(a, b); }
    __device__ __forceinline__ float identity() const { return -FLT_MAX; }
    __device__ __forceinline__ void atomic_op(float* address, float val) const {
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }
};

template <>
struct MaxOp<half> {
    __device__ __forceinline__ half operator()(const half& a, const half& b) const { 
        return __hgt(a, b) ? a : b; 
    }
    
    // half2 SIMD max
    __device__ __forceinline__ half2 op2(const half2& a, const half2& b) const { 
        return __hmax2(a, b); 
    }
    
    __device__ __forceinline__ half fold(const half2& a) const { 
        return __hgt(a.x, a.y) ? a.x : a.y; 
    }

    __device__ __forceinline__ half identity() const { 
        return __ushort_as_half(0xFC00); // 绝对安全的 half -inf
    }
    __device__ __forceinline__ half2 identity2() const { 
        half inf = __ushort_as_half(0xFC00);
        return __halves2half2(inf, inf); 
    }

    __device__ __forceinline__ void atomic_op(half* address, half val) const {
        unsigned short* address_as_ushort = (unsigned short*)address;
        unsigned short old = *address_as_ushort, assumed;
        do {
            assumed = old;
            half assumed_half = __ushort_as_half(assumed);
            half max_val = __hgt(val, assumed_half) ? val : assumed_half;
            old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(max_val));
        } while (assumed != old);
    }
};


// Kernel 逻辑
template <typename T, typename ReduceOp>
__device__ __forceinline__ T warp_reduce(T val, ReduceOp op) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = op(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// use Grid-Stride Loop arch to reduce atomic conflict
template <typename T, typename T_OUT, typename ReduceOp>
__global__ void block_reduce(
    const uint32_t n_vector_loads,
    const ReduceOp reduce_op,
    const T* __restrict__ input,
    T_OUT* __restrict__ output)
{
    // 一个 Stride 就是 Grid 的线程数量
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    __shared__ T_OUT sdata[32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    using T_DECAYED = std::decay_t<T>;
    T_OUT val = reduce_op.identity();

    // 向量化输入指针，并提示编译器内存是对齐的
    const float4* vec_input = reinterpret_cast<const float4*>(__builtin_assume_aligned(input, 16));

    if constexpr (std::is_same_v<T_DECAYED, float>) {
        // Grid-Stride Loop
        for (uint32_t i = tid; i < n_vector_loads; i += stride) {
            float4 vals = vec_input[i];
            T_OUT local_val = reduce_op(vals.x, vals.y);
            local_val = reduce_op(local_val, vals.z);
            local_val = reduce_op(local_val, vals.w);
            val = reduce_op(val, local_val);
        }
    } 
    else if constexpr (std::is_same_v<T_DECAYED, half>) {
        half2 val2 = reduce_op.identity2(); // 维护一个 half2 状态

        for (uint32_t i = tid; i < n_vector_loads; i += stride) {
            float4 f4_vals = vec_input[i];
            
            // 将 16 Bytes (float4) 强转为 4 个 half2
            const half2* h2_vals = reinterpret_cast<const half2*>(&f4_vals);

            // SIMD 极速归约
            half2 local_val2 = h2_vals[0];
            #pragma unroll
            for (int j = 1; j < 4; j++) {
                local_val2 = reduce_op.op2(local_val2, h2_vals[j]);
            }
            val2 = reduce_op.op2(val2, local_val2);
        }
        // 循环结束后，将 half2 折叠成 half
        val = reduce_op(val, reduce_op.fold(val2));
    }

    // Warp 级规约
    val = warp_reduce(val, reduce_op);

    // Block 级规约
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / warpSize)) ? sdata[lane] : reduce_op.identity();
        val = warp_reduce(val, reduce_op);

        // 唯一一次原子写（仅由每个 Block 的 0 号线程执行）
        if (lane == 0) {
            reduce_op.atomic_op(output, val);
        }
    }
}


template <typename T, const int BLOCK_SIZE = 256>
void reduce_sum_c(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);
    assert(n_elements % N_ELEMS_PER_LOAD == 0);
    uint32_t n_vector_loads = n_elements / N_ELEMS_PER_LOAD;

    // Grid-Stride 启动策略：最多起 1024 个 Block（例如 108个SM的 A100，能充分填满，且 Atomic 冲突小）
    uint32_t blocks = std::min((n_vector_loads + BLOCK_SIZE - 1) / BLOCK_SIZE, (uint32_t)1024);

    block_reduce<T, T, SumOp<T>><<<blocks, BLOCK_SIZE, 0, stream>>>(n_vector_loads, SumOp<T>(), d_input, d_output);
}

template <typename T, const int BLOCK_SIZE = 256>
void reduce_max_c(T *d_input, T *d_output, int n_elements, cudaStream_t stream = 0)
{
    const uint32_t N_ELEMS_PER_LOAD = 16 / sizeof(T);
    assert(n_elements % N_ELEMS_PER_LOAD == 0);
    uint32_t n_vector_loads = n_elements / N_ELEMS_PER_LOAD;

    uint32_t blocks = std::min((n_vector_loads + BLOCK_SIZE - 1) / BLOCK_SIZE, (uint32_t)1024);

    block_reduce<T, T, MaxOp<T>><<<blocks, BLOCK_SIZE, 0, stream>>>(n_vector_loads, MaxOp<T>(), d_input, d_output);
}

} // namespace fray