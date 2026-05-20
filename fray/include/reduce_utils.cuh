#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cute/tensor.hpp> 

namespace fray {
namespace utils {

// Traits for Numeric Limits
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<float> {
    __device__ __forceinline__ static float lowest() { 
        return -CUDART_INF_F; // -Infinity
    }
    __device__ __forceinline__ static float max() { 
        return CUDART_INF_F;  // +Infinity
    }
};

template <>
struct NumericLimits<half> {
    __device__ __forceinline__ static half lowest() { 
        // 浮点数强转，编译器会优化为常量
        return __float2half(-CUDART_INF_F); 
    }
    __device__ __forceinline__ static half max() { 
        return __float2half(CUDART_INF_F); 
    }
};

template <>
struct NumericLimits<int> {
    __device__ __forceinline__ static int lowest() { 
        return -2147483648; // 0x80000000
    }
    __device__ __forceinline__ static int max() { 
        return 2147483647;  // 0x7FFFFFFF
    }
};

// Reduce Op
template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
    __device__ __forceinline__ T identity() const { return T(0); }
    __device__ __forceinline__ void atomic_op(T* address, T val) const { atomicAdd(address, val); }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a > b)? a : b; }
    __device__ __forceinline__ T identity() const { return NumericLimits<T>::lowest();}
    __device__ __forceinline__ void atomic_op(T* address, T val) const { atomic_max_generic(address, val); }


private:
    __device__ __forceinline__ void atomic_max_generic(int* address,  int val) const {
        atomicMax(address, val);
    }
    __device__ __forceinline__ void atomic_max_generic(float* address, float val) const {
        int* address_as_int = reinterpret_cast<int*>(address);
        int old = *address_as_int;
        int assumed;
        do{
            assumed = old;
            old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while(assumed != old);
    }
    __device__ __forceinline__ void atomic_max_generic(half* address, half val) const {
        unsigned short* address_as_ushort = reinterpret_cast<unsigned short*>(address);
        unsigned short old = *address_as_ushort;
        unsigned short assumed;
        do {
            assumed = old;
            half old_h = __ushort_as_half(assumed);
            half max_h = (val > old_h) ? val : old_h;
            old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(max_h));
        } while (assumed != old);
    }
};



// Device Reduce Functions
// N threads reduce
template <int N = 32, typename T, typename ReduceOp>
__device__ T warp_reduce(T val, ReduceOp op){
    CUTE_UNROLL
    for(int offset = N / 2;offset > 0; offset >>=1){
        val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T, typename ReduceOp>
__device__ T block_reduce(T val, ReduceOp op){
    __shared__ T s[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce(val, op);

    if(lane_id == 0){
        s[warp_id] = val;
    }
    __syncthreads();

    if(warp_id == 0){
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? s[lane_id] : op.identity();
        val = warp_reduce(val, op);
    }
    __syncthreads(); 
    return val;
}

} // namespace utils
} // namespace fray