#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include "reduce_utils.cuh"
#include "utils.cuh"


namespace fray{

using namespace cute;
using namespace fray::utils;
using Element = cute::half_t;

struct DecodeArgs {
    const Element* Q;
    const Element* K;
    const Element* V;
    float* O_tmp;      
    float* m_tmp;       
    float* l_tmp;      
    float scale;
    int num_blocks;  
    int seq_len;
    int H_q;
    int H_kv;
    int D;
};

template <int G = 4, int Bc = 128, int D = 128>
__global__ void flash_decode_stage1_simt(DecodeArgs args) 
{
    // Grid.x = num_blocks
    // Grid.y = Batch * H_kv
    int chunk_idx = blockIdx.x; 
    int b_hkv_idx = blockIdx.y;

    int b = b_hkv_idx / args.H_kv; // Batch
    int h_kv = b_hkv_idx % args.H_kv; // head_kv
    int h_q_start = h_kv * G; // q_start_idx

    int q_offset = (b * args.H_q + h_q_start) * D;
    int kv_offset = (b * args.H_kv + h_kv) * args.seq_len * D + chunk_idx * Bc * D;

    // Workspace offset [B, H_q, NumBlocks, D]
    int out_offset = (b * args.H_q + h_q_start) * args.num_blocks * D + chunk_idx * D;
    int stat_offset = (b * args.H_q + h_q_start) * args.num_blocks + chunk_idx;


    Tensor mQ = make_tensor(make_gmem_ptr(args.Q + q_offset), make_shape(Int<G>{}, Int<D>{}), make_stride(Int<D>{}, _1{}));
    Tensor mK = make_tensor(make_gmem_ptr(args.K + kv_offset), make_shape(Int<Bc>{}, Int<D>{}), make_stride(Int<D>{}, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(args.V + kv_offset), make_shape(Int<Bc>{}, Int<D>{}), make_stride(Int<D>{}, _1{}));

    extern __shared__ char smem_raw[];
    Element* sq_data = reinterpret_cast<Element*>(smem_raw);
    Element* skv_data = sq_data + G * D;

    using Swizzle_333 = Swizzle<3, 3, 3>;
    auto sq_layout  = composition(Swizzle_333{}, make_layout(make_shape(Int<G>{}, Int<D>{}), make_stride(Int<D>{}, _1{})));
    auto skv_layout = composition(Swizzle_333{}, make_layout(make_shape(Int<Bc>{}, Int<D>{}), make_stride(Int<D>{}, _1{})));
    auto svt_layout = composition(skv_layout, make_layout(make_shape(Int<D>{}, Int<Bc>{}), GenRowMajor{}));
    auto sp_layout  = composition(Swizzle_333{}, make_layout(make_shape(Int<G>{}, Int<Bc>{}), make_stride(Int<Bc>{}, _1{})));

    Tensor sQ = make_tensor(make_smem_ptr(sq_data), sq_layout);
    Tensor sK = make_tensor(make_smem_ptr(skv_data), skv_layout);
    Tensor sV = make_tensor(make_smem_ptr(skv_data), skv_layout);
    Tensor sP = make_tensor(make_smem_ptr(sq_data), sp_layout);

    auto copy_Q = make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{}, 
                                  Layout<Shape<Int<G>, _32>, Stride<_32, _1>>{}, Layout<Shape<_1, _4>>{});
    auto thr_copy_Q = copy_Q.get_thread_slice(threadIdx.x);
    auto copy_KV = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{}, 
                                   Layout<Shape<Int<G * 32 / 4>, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _8>>{}); 
    auto thr_copy_KV = copy_KV.get_thread_slice(threadIdx.x);

    // Load Q&K from g2s
    Tensor tQgQ = thr_copy_Q.partition_S(mQ);
    Tensor tQsQ = thr_copy_Q.partition_D(sQ);
    Tensor tKgK = thr_copy_KV.partition_S(mK);
    Tensor tKsK = thr_copy_KV.partition_D(sK);
    copy(copy_Q, tQgQ, tQsQ);
    copy(copy_KV, tKgK, tKsK);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    using MMA_Atom = UniversalFMA<float, Element, Element, float>;
    auto tiled_mma = make_tiled_mma(MMA_Atom{}, 
                                make_layout(make_shape(Int<G>{}, _32{}, _1{}), 
                                            make_stride(_32{}, _1{}, _0{})));
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // S = Q * K^T
    Tensor tSsQ = thr_mma.partition_A(sQ); // [G, D]
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ); 
    Tensor tSsK = thr_mma.partition_B(sK); // [Bc, D]
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    copy(tSsQ, tSrQ);
    copy(tSsK, tSrK);

    Tensor tSrS = partition_fragment_C(tiled_mma, make_shape(Int<G>{}, Int<Bc>{}));
    clear(tSrS);

    gemm(tiled_mma, tSrS, tSrQ, tSrK, tSrS);

    float local_max = -INFINITY;
    CUTE_UNROLL
    for(int i = 0; i < size<2>(tSrS); ++i) {
        local_max = fmaxf(local_max, tSrS(0, 0, i) * args.scale);
    }
    float warp_max = warp_reduce<32>(local_max, MaxOp<float>{});

    float local_sum = 0.0f;
    Tensor tSrP = make_tensor_like<Element>(tSrS);
    CUTE_UNROLL
    for(int i = 0; i < size<2>(tSrS); ++i) {
        int k_idx = chunk_idx * Bc + (threadIdx.x % 32) + i * 32;
        float p = 0.0f;
        if (k_idx < args.seq_len) {
            p = expf(float(tSrS(0, 0, i)) * args.scale - warp_max);
        }
        tSrP(0, 0, i) = Element(p); 
        local_sum += p;
    }
    float warp_sum = warp_reduce<32>(local_sum, SumOp<float>{});

    // FP32 to FP16 & write P to share
    Tensor tSsP = thr_mma.partition_C(sP);
    copy(tSrP, tSsP); 
    __syncthreads();

    // Load V
    __syncthreads();
    copy(copy_KV, thr_copy_KV.partition_S(mV), thr_copy_KV.partition_D(sV));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Load P from s to r
    Tensor tOsP = thr_mma.partition_A(sP);
    Tensor tOrP = thr_mma.partition_fragment_A(sP);
    copy(tOsP, tOrP);

    // O = P * V ---
    auto sVt = make_tensor(sV.data(), svt_layout);
    Tensor tOsVt = thr_mma.partition_B(sVt);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt);
    copy(tOsVt, tOrVt);
    
    Tensor tOrO = partition_fragment_C(tiled_mma, make_shape(Int<G>{}, Int<D>{}));
    clear(tOrO);

    gemm(tiled_mma, tOrO, tOrP, tOrVt, tOrO);

    // Write to workspace for reduce in Stage2
    Tensor mO_tmp = make_tensor(make_gmem_ptr(args.O_tmp + out_offset), make_shape(Int<G>{}, Int<D>{}), make_stride(args.num_blocks * Int<D>{}, _1{}));
    Tensor tOgO = thr_mma.partition_C(mO_tmp);
    copy(tOrO, tOgO);

    if (threadIdx.x % 32 == 0) {
        int warp_id = threadIdx.x / 32;
        args.m_tmp[stat_offset + warp_id * args.num_blocks] = warp_max;
        args.l_tmp[stat_offset + warp_id * args.num_blocks] = warp_sum;
    }
}


template <int D>
__global__ void flash_decode_stage2_kernel(
    float* __restrict__ O_tmp,  // [B, H_q, num_blocks, D]
    float* __restrict__ m_tmp,  // [B, H_q, num_blocks]
    float* __restrict__ l_tmp,  // [B, H_q, num_blocks]
    Element*  __restrict__ O,      // [B, H_q, D]
    int num_blocks,
    int H_q) 
{
    // 一个 Block 负责一个头的规约
    // Grid.x = H_q, Grid.y = Batch
    int h_q = blockIdx.x;
    int b   = blockIdx.y;
    int tid = threadIdx.x; // [0, D-1] 负责特定的 D 维度特征

    // 计算当前 Query 对应的偏移量
    int stat_base = (b * H_q + h_q) * num_blocks;
    int out_base  = stat_base * D;

    float* m_ptr = m_tmp + stat_base;
    float* l_ptr = l_tmp + stat_base;
    float* o_tmp_ptr = O_tmp + out_base;
    Element*  o_ptr = O + (b * H_q + h_q) * D;

    // 利用 Shared Memory 缓存 m 和 l，避免后续被重复读取 D 次
    // num_blocks * 2 * sizeof(float)
    extern __shared__ float smem[]; 
    float* smem_m = smem;
    float* smem_l = smem + num_blocks;

    // 协作加载 m_tmp，并求出全局最大值 Global_M
    float thread_m = -INFINITY;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float m_val = m_ptr[i];
        smem_m[i] = m_val;
        thread_m = fmaxf(thread_m, m_val);
    }
    
    float global_m = block_reduce<float>(thread_m, MaxOp<float>{});
    
    // 广播 global_m 给所有线程
    __shared__ float s_global_m;
    if (tid == 0) s_global_m = global_m;
    __syncthreads();
    global_m = s_global_m;

    // 协作加载 l_tmp，并求出全局指数和 Global_L
    float thread_l_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float l_val = l_ptr[i];
        smem_l[i] = l_val; // 存进共享内存缓存
        
        // 公式：L_new = L_old * exp(m_old - m_global)
        thread_l_sum += l_val * expf(smem_m[i] - global_m);
    }

    float global_l = block_reduce<float>(thread_l_sum, SumOp<float>{});
    
    __shared__ float s_global_l;
    if (tid == 0) s_global_l = global_l;
    __syncthreads();
    global_l = s_global_l;

    // 每个线程独立合并自己的 O 维度特征
    float inv_global_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
    float o_acc = 0.0f;

    for (int i = 0; i < num_blocks; ++i) {
        // O_new += O_old * exp(m_old - m_global) / L_global
        float scale = expf(smem_m[i] - global_m) * inv_global_l;
        o_acc += o_tmp_ptr[i * D + tid] * scale;
    }

    if (tid < D) {
        o_ptr[tid] = __float2half(o_acc);
    }
}



template <int G = 4, int Bc = 128, int D = 128>
void flash_decoding_cute(
    Element* Q, Element* K, Element* V, Element* O,
    void* workspace, // 在Python分配一大块连续物理内存
    int batch_size, int num_heads, int num_kv_heads, int seq_len, 
    cudaStream_t stream) 
{
    int num_blocks = (seq_len + Bc - 1) / Bc;
    float scale = 1.0f / sqrt((float)D);

    // 在 C++ 内部切分 Workspace 指针
    float* O_tmp = reinterpret_cast<float*>(workspace);
    float* m_tmp = O_tmp + (batch_size * num_heads * num_blocks * D);
    float* l_tmp = m_tmp + (batch_size * num_heads * num_blocks);

    // Stage 1 (局部 Attention)
    DecodeArgs args = {
        Q, K, V, O_tmp, m_tmp, l_tmp, scale, 
        num_blocks, seq_len, num_heads, num_kv_heads, D
    };

    dim3 grid1(num_blocks, batch_size * num_kv_heads);
    dim3 block1(G * 32); // 一个warp负责一个Q头
    int smem_size1 = (G * D + Bc * D) * sizeof(Element);
    
    flash_decode_stage1_simt<G, Bc, D><<<grid1, block1, smem_size1, stream>>>(args);

    // Stage 2 (全局规约)
    dim3 grid2(num_heads, batch_size);
    dim3 block2(D); // 每个线程负责一个 D 维特征
    int smem_size2 = num_blocks * 2 * sizeof(float);

    flash_decode_stage2_kernel<D><<<grid2, block2, smem_size2, stream>>>(
        O_tmp, m_tmp, l_tmp, O, num_blocks, num_heads);
    
}

} // namespace fray