#pragma once

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/numeric/math.hpp>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "softmax.cuh"

namespace fray{

using namespace cute;
using Element = half;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <class Layout>
__forceinline__ __device__ auto convert_reg_layout_c2a(Layout layout)
{
    using namespace cute;
    auto l = logical_divide(layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

template <typename ShapeT, typename CtaTiler,
          typename SmemLayoutQ, typename SmemLayoutKV,typename SmemLayoutVt,
          typename TiledMma0, typename TiledMma1,
          typename TiledCopyQKV_g2s, 
          typename TiledCopyQ_s2r,typename TiledCopyK_s2r, typename TiledCopyVt_s2r,
          typename TiledCopyO_r2s, typename TiledCopyO_s2g>
__global__ void flash_attn_cute_kernel(
    Element* Q, Element* K, Element* V, Element* O,
    float scale,
    ShapeT tensor_shape, CtaTiler cta_tiler,
    SmemLayoutQ sQ_layout, SmemLayoutKV sKV_layout,SmemLayoutVt sVt_layout, 
    TiledMma0 tiled_mma0, TiledMma1 tiled_mma1,
    TiledCopyQKV_g2s copy_g2s, 
    TiledCopyQ_s2r copyQ_s2r, TiledCopyK_s2r copyK_s2r, TiledCopyVt_s2r copyVt_s2r,
    TiledCopyO_r2s copyO_r2s, TiledCopyO_s2g copyO_s2g)
{
    int head_idx = blockIdx.y % size<1>(tensor_shape);
    int seq_block_idx = blockIdx.x;
    int64_t offset = (blockIdx.y) * size<2>(tensor_shape) * size<3>(tensor_shape);
    Element* q_ptr = Q + offset;
    Element* k_ptr = K + offset;
    Element* v_ptr = V + offset;
    Element* o_ptr = O + offset; 

    // Global Tensor：[seq_len, head_dim]
    auto g_shape  = make_shape(size<2>(tensor_shape), size<3>(tensor_shape));
    auto g_stride = make_stride(size<3>(tensor_shape), _1{}); // 行主序
    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr), g_shape, g_stride);
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr), g_shape, g_stride);
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr), g_shape, g_stride);
    Tensor mO = make_tensor(make_gmem_ptr(o_ptr), g_shape, g_stride);

    // 切出当前 Block 负责的 Q 和 O 的 Tile: [Br, Bd]
    auto tile_shape_QO = make_shape(size<0>(cta_tiler), size<2>(cta_tiler)); // [Br, Bd]
    auto tile_shape_KV = make_shape(size<1>(cta_tiler), size<2>(cta_tiler)); // [Bc, Bd]

    Tensor gQ = local_tile(mQ, tile_shape_QO, make_coord(seq_block_idx, 0)); 
    Tensor gO = local_tile(mO, tile_shape_QO, make_coord(seq_block_idx, 0));

    // Shared Memory
    extern __shared__ char smem_raw[];
    Element* sq_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sk_ptr = sq_ptr + cosize(sQ_layout);
    Element* sv_ptr = sk_ptr + cosize(sKV_layout);

    Tensor sQ = make_tensor(make_smem_ptr(sq_ptr), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(sk_ptr), sKV_layout);

    Tensor sV = make_tensor(make_smem_ptr(sv_ptr), sKV_layout);
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVt{}); 
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), get_nonswizzle_portion(SmemLayoutVt{}));

    // Thread Partition
    auto thr_mma0 = tiled_mma0.get_thread_slice(threadIdx.x);
    auto thr_mma1 = tiled_mma1.get_thread_slice(threadIdx.x);
    auto thr_copy = copy_g2s.get_thread_slice(threadIdx.x);

    // Init O Accumulator (FP32)
    Tensor tOrO = thr_mma1.partition_fragment_C(gO); // [MMA, MMA_M, MMA_K]
    clear(tOrO);

    // Online softmax States
    static constexpr int kRowsPerThread = decltype(size<1>(tOrO))::value * 2;
    OnlineSoftmax<kRowsPerThread> softmax;
    float scale_log2 = scale * M_LOG2E; // scale * log2(e)

    // Load Q from global to shared
    auto tQgQ = thr_copy.partition_S(gQ);
    auto tQsQ = thr_copy.partition_D(sQ);
    copy(copy_g2s, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Load Q from shared to reg
    auto thr_copy_q_s2r = copyQ_s2r.get_thread_slice(threadIdx.x);
    auto tSsQ = thr_copy_q_s2r.partition_S(sQ);
    auto tSrQ = thr_mma0.partition_fragment_A(sQ); 
    auto tSrQ_view = thr_copy_q_s2r.retile_D(tSrQ);
    copy(copyQ_s2r, tSsQ, tSrQ_view);

    // Main Loop
    int num_k_blocks = (int(size<2>(tensor_shape)) + int(size<1>(cta_tiler)) - 1) / int(size<1>(cta_tiler));
    for(int j = 0; j < num_k_blocks; ++j){
        // Load K,V from global to shared
        auto gK = local_tile(mK, tile_shape_KV, make_coord(j, 0)); // [Bc, Bd]
        auto gV = local_tile(mV, tile_shape_KV, make_coord(j, 0)); // [Bc, Bd]
        auto tKgK = thr_copy.partition_S(gK);
        auto tKsK = thr_copy.partition_D(sK);
        auto tVgV = thr_copy.partition_S(gV);
        auto tVsV = thr_copy.partition_D(sV);
        copy(copy_g2s, tKgK, tKsK);
        copy(copy_g2s, tVgV, tVsV);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // MMA0 : S = Q * K^T
        auto thr_copy_k_s2r = copyK_s2r.get_thread_slice(threadIdx.x);
        auto tSsK = thr_copy_k_s2r.partition_S(sK);
        auto tSrK = thr_mma0.partition_fragment_B(sK);
        auto tSrK_view = thr_copy_k_s2r.retile_D(tSrK);
        copy(copyK_s2r, tSsK, tSrK_view);

        Tensor tSrS = thr_mma0.partition_fragment_C(make_tensor<Element>(make_shape(size<0>(cta_tiler), size<1>(cta_tiler))));// [Br, Bc]
        clear(tSrS);
        gemm(tiled_mma0, tSrS, tSrQ, tSrK, tSrS);

        softmax.update(tSrS, tOrO, scale_log2);

        // MMA1：O += P * V
        Tensor rP = convert_type<Element>(tSrS);
        Tensor tOrP = make_tensor(rP.data(), convert_reg_layout_c2a(rP.layout()));

        auto thr_copy_vt_s2r = copyVt_s2r.get_thread_slice(threadIdx.x);
        auto tOsVt = thr_copy_vt_s2r.partition_S(sVt);
        auto tOrVt = thr_mma1.partition_fragment_B(sVtNoSwizzle);
        auto tOrVt_view = thr_copy_vt_s2r.retile_D(tOrVt);
        copy(copyVt_s2r, tOsVt, tOrVt_view);

        gemm(tiled_mma1, tOrO, tOrP, tOrVt, tOrO);
        __syncthreads();
    }

    // Normalize O
    softmax.finalize(tOrO);

    // Write back O
    // Write O from reg to shared
    __syncthreads();
    Tensor sO = make_tensor(make_smem_ptr(sq_ptr), sQ_layout);

    auto thr_copy_o_r2s =  copyO_r2s.get_thread_slice(threadIdx.x);
    Tensor tOrO_r2s = thr_copy_o_r2s.retile_S(tOrO);
    Tensor tOsO_r2s = thr_copy_o_r2s.partition_D(sO);

    auto thr_copy_o_s2g = copyO_s2g.get_thread_slice(threadIdx.x);
    Tensor tOsO_s2g = thr_copy_o_s2g.partition_S(sO);
    Tensor tOgO_s2g = thr_copy_o_s2g.partition_D(gO);

    Tensor tOrO_r2sx = group_modes<1, 3>(tOrO_r2s);
    Tensor tOsO_r2sx = group_modes<1, 3>(tOsO_r2s); 
    Tensor tOsO_s2gx = group_modes<1, 3>(tOsO_s2g); 
    Tensor tOgO_s2gx = group_modes<1, 3>(tOgO_s2g);

    CUTE_UNROLL
    for (int i = 0; i < size<1>(tOrO_r2sx); ++i) {
        // 创建连续的临时寄存器 t (同时隐式完成 FP32 到 FP16 的转换)
        auto t = make_tensor_like<Element>(tOrO_r2sx(_, i));
        copy(tOrO_r2sx(_, i), t); 
        copy(copyO_r2s, t, tOsO_r2sx(_, i));
    }
    
    // 关键同步：必须等所有线程把 O 全部写进 Smem
    __syncthreads();

    // 将 O 从共享内存利用 128-bit 向量化合并写入全局内存 (Smem -> Gmem)
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tOsO_s2gx); ++i) {
        copy(copyO_s2g, tOsO_s2gx(_, i), tOgO_s2gx(_, i));
    }
}


template <int HEAD_DIM = 128>
void flash_attn_cute(Element* query, Element* key, Element* value, half *output, int batch_size, int num_heads, int seq_len, cudaStream_t stream)
{
    // 先假设QKV对应的的序列长度一样，均为sep_len
    // key: [batch_size, num_heads, seq_len, head_dim]
    // value: [batch_size, num_heads, seq_len, head_dim]
    // query: [batch_size, num_heads, seq_len, head_dim]
    // output: [batch_size, num_heads, seq_len, head_dim]
    // scores/weights: [batch_size, num_heads, seq_len, seq_len] , weights = softmax(scores)

    auto tensor_shape = make_shape(batch_size, num_heads, seq_len, Int<HEAD_DIM>{});
    // 每个block负责计算最终结果[Br, HEAD_DIM] = [128, 128]的输出
    auto Br = Int<128>{};
    auto Bc = Int<32>{};
    auto Bd = Int<HEAD_DIM>{};
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
    // MMA0: 用于计算 S = Q * K^T. 
    // Q 尺寸为 [128, 128], K 尺寸为 [32, 128], 输出 S 为 [128, 32]
    TiledMMA tiled_mma0 = make_tiled_mma(MMA_Atom{},
                                         Layout<Shape<_4, _1, _1>>{},  // 128 线程: 4 warps in M, 1 warp in N
                                         Tile<_128, _32, _16>{});      // Tile 覆盖整个 S 矩阵的分块大小

    // MMA1: 用于计算 O = P * V. 
    // P 尺寸为 [128, 32], V 尺寸为 [32, 128], 输出 O 为 [128, 128]
    TiledMMA tiled_mma1 = make_tiled_mma(MMA_Atom{},
                                         Layout<Shape<_4, _1, _1>>{},
                                         Tile<_128, _128, _32>{});     // Tile 覆盖整个 O 矩阵的分块大小

    // Define Copies
    // Global To Shared
    TiledCopy copyQKV_g2s = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // 16x8 = 128 线程
        Layout<Shape<_1, _8>>{});                 // 每个线程搬 8 个元素
    // Shared To Reg
    TiledCopy copyQ_s2r = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma0);
    TiledCopy copyK_s2r = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma0);
    TiledCopy copyVt_s2r = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, tiled_mma1);

    TiledCopy copyO_r2s = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, Element>{}, tiled_mma1);
    TiledCopy copyO_s2g = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{}, 
        Layout<Shape<_16, _8>, Stride<_8, _1>>{}, 
        Layout<Shape<_1, _8>>{});

    int seq_blocks = (seq_len + Br - 1) / Br;    
    float scale = 1.0f / sqrt((float)HEAD_DIM); 

    // Grid.x 负责序列长度分块, Grid.y 负责 Batch 和 Heads                            
    dim3 dimGrid(seq_blocks, batch_size * num_heads);
    dim3 dimBlock(int(size(tiled_mma0))); // 128 线程

    static constexpr int smem_size = (cosize(sQ_layout) + cosize(sKV_layout) * 2) * sizeof(Element);

    auto kernel = flash_attn_cute_kernel<
        decltype(tensor_shape), decltype(cta_tiler), 
        decltype(sQ_layout), decltype(sKV_layout), decltype(sVt_layout),
        decltype(tiled_mma0), decltype(tiled_mma1),
        decltype(copyQKV_g2s), decltype(copyQ_s2r), 
        decltype(copyK_s2r), decltype(copyVt_s2r),
        decltype(copyO_r2s), decltype(copyO_s2g)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        query, key, value, output,
        scale, tensor_shape, cta_tiler,
        sQ_layout, sKV_layout, sVt_layout,
        tiled_mma0, tiled_mma1,
        copyQKV_g2s, copyQ_s2r, copyK_s2r, copyVt_s2r,
        copyO_r2s, copyO_s2g);
}

} // namespace fray