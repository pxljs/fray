#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include "softmax.cuh"

namespace fray{

using namespace cute;
using Element = half;

template <typename To, typename Engine, typename Layout>
__device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To, From, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From, numel> *>(tensor.data())); // frag:cutlass::Array<To, numel>
    return make_tensor(make_rmem_ptr<To>(&frag), tensor.layout());
}

template <typename Layout>
__device__ auto convert_reg_layout_c2a(Layout layout) {
    auto l = logical_divide(layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

template <typename ShapeT, typename CtaTiler,
          typename SmemLayoutQ, typename SmemLayoutKV, typename SmemLayoutVt,
          typename TiledMma, 
          typename TiledCopyG2S, 
          typename TiledCopyQ_S2R, typename TiledCopyK_S2R, typename TiledCopyVt_S2R,
          typename TiledCopyO_R2S, typename TiledCopyO_S2G>
__global__ void flash_attn_cute_kernel(
    Element* Q, Element* K, Element* V, Element* O,
    float scale, ShapeT tensor_shape, CtaTiler cta_tiler,
    SmemLayoutQ sQ_layout, SmemLayoutKV sKV_layout, SmemLayoutVt sVt_layout, 
    TiledMma tiled_mma,
    TiledCopyG2S copy_g2s, 
    TiledCopyQ_S2R copyQ_s2r, TiledCopyK_S2R copyK_s2r, TiledCopyVt_S2R copyVt_s2r,
    TiledCopyO_R2S copyO_r2s, TiledCopyO_S2G copyO_s2g) 
{
    int64_t head_offset = int64_t(blockIdx.y) * size<2>(tensor_shape) * size<3>(tensor_shape);
    auto g_shape  = make_shape(size<2>(tensor_shape), size<3>(tensor_shape));
    auto g_stride = make_stride(size<3>(tensor_shape), _1{});

    // Global Q/K/V/O
    Tensor mQ = make_tensor(make_gmem_ptr(Q + head_offset), g_shape, g_stride);
    Tensor mK = make_tensor(make_gmem_ptr(K + head_offset), g_shape, g_stride);
    Tensor mV = make_tensor(make_gmem_ptr(V + head_offset), g_shape, g_stride);
    Tensor mO = make_tensor(make_gmem_ptr(O + head_offset), g_shape, g_stride);

    // Tile partition for block
    auto tile_QO = make_shape(size<0>(cta_tiler), size<2>(cta_tiler)); // [Br, Bd]
    auto tile_KV = make_shape(size<1>(cta_tiler), size<2>(cta_tiler)); // [Bc, Bd]
    Tensor gQ = local_tile(mQ, tile_QO, make_coord(blockIdx.x, 0));  // [Br, Bd, Tr]
    Tensor gO = local_tile(mO, tile_QO, make_coord(blockIdx.x, 0));  // [Br, Bd, Tr]
    Tensor gK = local_tile(mK, tile_KV, make_coord(_, 0)); // [Bc, Bd, Tc]
    Tensor gV = local_tile(mV, tile_KV, make_coord(_, 0)); // [Bc, Bd, Tc]

    // Shared Memory
    extern __shared__ char smem_raw[];
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_raw)), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(sQ.data().get() + cosize(sQ_layout)), sKV_layout);
    Tensor sV = make_tensor(make_smem_ptr(sK.data().get() + cosize(sKV_layout)), sKV_layout);
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVt{}); 
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), get_nonswizzle_portion(SmemLayoutVt{}));

    // Define thread g2s_copy_qkv
    ThrCopy thr_g2s  = copy_g2s.get_thread_slice(threadIdx.x);
    Tensor tQgQ = thr_g2s.partition_S(gQ);
    Tensor tQsQ = thr_g2s.partition_D(sQ);

    Tensor tKgK = thr_g2s.partition_S(gK);
    Tensor tKsK = thr_g2s.partition_D(sK);

    Tensor tVgV = thr_g2s.partition_S(gV);
    Tensor tVsV = thr_g2s.partition_D(sV);

    // Define thread s2r_copy k/q/v
    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    ThrCopy thr_copy_q_s2r = copyQ_s2r.get_thread_slice(threadIdx.x);
    Tensor tSsQ = thr_copy_q_s2r.partition_S(sQ);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ); 
    Tensor tSrQ_view = thr_copy_q_s2r.retile_D(tSrQ);

    ThrCopy thr_copy_k_s2r = copyK_s2r.get_thread_slice(threadIdx.x);
    Tensor tSsK = thr_copy_k_s2r.partition_S(sK);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tSrK_view = thr_copy_k_s2r.retile_D(tSrK);

    ThrCopy thr_copy_vt_s2r = copyVt_s2r.get_thread_slice(threadIdx.x);
    Tensor tOsVt = thr_copy_vt_s2r.partition_S(sVt);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tOrVt_view = thr_copy_vt_s2r.retile_D(tOrVt);

    // Load Q from global to shared
    copy(copy_g2s, tQgQ, tQsQ);
    // Preload the first k tile from global to shared
    copy(copy_g2s, tKgK(_,_,_,0), tKsK);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Load Q from shared to reg
    copy(copyQ_s2r, tSsQ, tSrQ_view);

    // Init S and O Accumulator (FP32)
    Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(cta_tiler));
    Tensor tOrO = thr_mma.partition_fragment_C(gO); // [MMA, MMA_M, MMA_K]
    clear(tOrO);

    // Online softmax States
    static constexpr int kRowsPerThread = decltype(size<1>(tOrO))::value * 2;
    OnlineSoftmax<kRowsPerThread> softmax;
    float scale_log2 = scale * 1.44269504f; // scale * log2(e)

    int num_kv_tiles = ceil_div(size<2>(tensor_shape), size<1>(cta_tiler));
    // Main Loop
    for(int j = 0; j < num_kv_tiles; ++j){
        // S = Q * K^T
        cp_async_wait<0>();
        __syncthreads();

        copy(copy_g2s, tVgV(_, _, _, j), tVsV(_, _, _));
        cp_async_fence();

        copy(copyK_s2r, tSsK, tSrK_view);
        __syncthreads();

        clear(tSrS);
        gemm(tiled_mma, tSrS, tSrQ, tSrK, tSrS);

        cp_async_wait<0>();
        __syncthreads();

        if (j < num_kv_tiles - 1) {
            copy(copy_g2s, tKgK(_,_,_, j+1), tKsK);
            cp_async_fence();
        }

        softmax.update(tSrS, tOrO, scale_log2);

        // O += P * V
        Tensor rP = convert_type<Element>(tSrS);
        Tensor tOrP = make_tensor(rP.data(), convert_reg_layout_c2a(rP.layout()));
        
        copy(copyVt_s2r, tOsVt, tOrVt_view);
        gemm(tiled_mma, tOrO, tOrP, tOrVt, tOrO);
    }
    // Normalize O
    softmax.finalize(tOrO);

    // Write back O
    // Write O from reg to shared
    __syncthreads();
    Tensor sO = make_tensor(sQ.data(), sQ_layout);
    auto thr_o_r2s = copyO_r2s.get_thread_slice(threadIdx.x);
    auto thr_o_s2g = copyO_s2g.get_thread_slice(threadIdx.x);

    // Reg -> Smem 
    auto tOrO_r2s = group_modes<1, 3>(thr_o_r2s.retile_S(tOrO));
    auto tOsO_r2s = group_modes<1, 3>(thr_o_r2s.partition_D(sO));
    for (int i = 0; i < size<1>(tOrO_r2s); ++i) {
        auto t_reg = make_tensor_like<Element>(tOrO_r2s(_, i));
        copy(tOrO_r2s(_, i), t_reg);
        copy(copyO_r2s, t_reg, tOsO_r2s(_, i));
    }

    __syncthreads();
    // Smem -> Global
    copy(copyO_s2g, thr_o_s2g.partition_S(sO), thr_o_s2g.partition_D(gO));
}


template <int R = 128, int C = 32,int D = 128>
void flash_attn_cute(Element* query, Element* key, Element* value, half *output, int batch_size, int num_heads, int seq_len, cudaStream_t stream)
{
    // 先假设QKV对应的的序列长度一样，均为sep_len
    // key: [batch_size, num_heads, seq_len, head_dim]
    // value: [batch_size, num_heads, seq_len, head_dim]
    // query: [batch_size, num_heads, seq_len, head_dim]
    // output: [batch_size, num_heads, seq_len, head_dim]
    // scores/weights: [batch_size, num_heads, seq_len, seq_len] , weights = softmax(scores)
    auto Br = Int<R>{};
    auto Bc = Int<C>{};
    auto Bd = Int<D>{};

    auto tensor_shape = make_shape(batch_size, num_heads, seq_len, Bd);
    // 每个block负责计算最终结果[Br, Bd] = [128, 128]的输出
    auto cta_tiler = make_shape(Br, Bc, Bd);

    // Define Layouts
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
            make_stride(Int<32>{}, Int<1>{}))));

    auto sQ_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Br, Bd));
    auto sKV_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(Bc, Bd));
    auto sVt_layout = composition(sKV_layout, make_layout(make_shape(Bd, Bc), GenRowMajor{}));

    // Define MMA
    using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
    TiledMMA tiled_mma = make_tiled_mma(MMA_Atom{},
                                         Layout<Shape<_4, _1, _1>>{},
                                         Tile<_64, _16, _16>{});

    // Define Copies
    // Global To Shared
    auto g2s_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                                    Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
    
    auto s2r_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    auto s2r_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    auto s2r_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, tiled_mma);

    TiledCopy r2s_copy_O = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, Element>{}, tiled_mma);
    TiledCopy s2g_copy_O = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{}, 
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, 
        Layout<Shape<_1, _8>>{});

    int seq_blocks = (seq_len + Br - 1) / Br;    
    float scale = 1.0f / sqrt((float)Bd); 

    // Grid.x 负责序列长度分块, Grid.y 负责 Batch 和 Heads                            
    dim3 dimGrid(seq_blocks, batch_size * num_heads);
    dim3 dimBlock(int(size(tiled_mma))); // 128 线程

    static constexpr int smem_size = (cosize(sQ_layout) + cosize(sKV_layout) * 2) * sizeof(Element);

    auto kernel = flash_attn_cute_kernel<
        decltype(tensor_shape), decltype(cta_tiler), 
        decltype(sQ_layout), decltype(sKV_layout), decltype(sVt_layout),
        decltype(tiled_mma), 
        decltype(g2s_copy), decltype(s2r_copy_Q), 
        decltype(s2r_copy_K), decltype(s2r_copy_V),
        decltype(r2s_copy_O), decltype(s2g_copy_O)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        query, key, value, output,
        scale, tensor_shape, cta_tiler,
        sQ_layout, sKV_layout, sVt_layout,
        tiled_mma, 
        g2s_copy, s2r_copy_Q, s2r_copy_K, s2r_copy_V,
        r2s_copy_O, s2g_copy_O);
}

} // namespace fray