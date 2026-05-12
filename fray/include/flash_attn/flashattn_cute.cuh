#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace fray{

using namespace cute;
using Element = half;

template <typename T>
__device__ __forceinline__ T quad_reduce_max(T val) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = max(val, __shfl_xor_sync(0xffffffff, val, 2));
    return val;
}

template <typename T>
__device__ __forceinline__ T quad_reduce_sum(T val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

template <typename ShapeT, typename CtaTiler,
          typename SmemLayoutQ, typename SmemLayoutKV,
          typename TiledMma0, typename TiledMma1,
          typename TiledCopyQKV_g2s, 
          typename TiledCopyQ_s2r,typename TiledCopyK_s2r, typename TiledCopyV_s2r>
__global__ void flash_attn_cute_kernell(
    Element* Q, Element* K, Element* V, Element* O,
    float scale,
    ShapeT tensor_shape, CtaTiler cta_tiler,
    SmemLayoutQ sQ_layout, SmemLayoutKV sKV_layout,
    TiledMma0 tiled_mma0, TiledMma1 tiled_mma1,
    TiledCopyQKV_g2s copy_g2s, 
    TiledCopyQ_s2r copyQ_s2r, TiledCopyK_s2r copyK_s2r, TiledCopyV_s2r copyV_s2r)
{
    int batch_idx = blockIdx.y / size<1>(tensor_shape);
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
    Tensor gQ = local_tile(mQ, cta_tiler, make_coord(seq_block_idx, _)); 
    Tensor gO = local_tile(mO, cta_tiler, make_coord(seq_block_idx, _));

    // Shared Memory
    extern __shared__ char smem_raw[];
    Element* sq_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sk_ptr = sq_ptr + cosize(sQ_layout);
    Element* sv_ptr = sk_ptr + cosize(sKV_layout);

    Tensor sQ = make_tensor(make_smem_ptr(sq_ptr), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(sk_ptr), sKV_layout);
    Tensor sV = make_tensor(make_smem_ptr(sv_ptr), sKV_layout);

    using SmemLayoutAtom = decltype(composition(Swizzle<3,3,3>{},
                                    make_layout(make_shape(Int<8>{}, Int<32>{}),
                                    make_stride(Int<32>{}, Int<1>{}))));
    auto sP_layout = tile_to_shape(SmemLayoutAtom{},
                                   make_shape(size<0>(cta_tiler),
                                   size<1>(cta_tiler))); // [Br, Bc]
    Tensor sP = make_tensor(make_smem_ptr(sk_ptr), sP_layout); // 因为 S = QK^T算完以后， K不需要了，复用K的共享内存区域

    // Thread Partition
    auto thr_mma0 = tiled_mma0.get_thread_slice(threadIdx.x);
    auto thr_mma1 = tiled_mma1.get_thread_slice(threadIdx.x);
    auto thr_copy = copy_g2s.get_thread_slice(threadIdx.x);

    // Init O Accumulator (FP32)
    Tensor tOrO = thr_mma1.partition_fragment_C(gO); // [MMA, MMA_M, MMA_K]
    clear(tOrO);

    // Online softmax States
    Tensor m = make_tensor<float>(make_shape(size<1>(tOrO))); 
    Tensor l = make_tensor<float>(make_shape(size<1>(tOrO))); 
    fill!(m, -INFINITY);
    fill!(l, 0.0f);

    // Load Q from global to shared
    auto tQgQ = thr_copy.partition_S(gQ);
    auto tQsQ = thr_copy.partition_D(sQ);
    copy(copy_g2s, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Load Q from shared to reg
    auto thr_copy_q_s2r = copyQ_s2r.get_thread_slice(threadIdx.x);
    auto tSsQ = thr_copy_q_s2r.partition_S(sQ)
    auto tSrQ = thr_mma0.partition_fragment_A(sQ); 
    auto tSrQ_view = thr_copy_q_s2r.retile_D(tSrQ);
    copy(copyQ_s2r, tQsQ, tSrQ_view);

    
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
    auto cta_tailer = make_shape(Br{}, Bc{}, Bd{});

    // Define Layouts
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
            make_stride(Int<32>{}, Int<1>{}))));

    auto sQ = tile_to_shape(SmemLayoutAtom{}, make_shape(Br, Bd));
    auto sKV = tile_to_shape(SmemLayoutAtom{}, make_shape(Bc, Bd));

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
    TiledCopy copyV_s2r = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma1);

    int seq_blocks = (seq_len + Br - 1) / Br;    
    float scale = 1.0f / sqrt((float)HEAD_DIM); 

    // Grid.x 负责序列长度分块, Grid.y 负责 Batch 和 Heads                            
    dim3 dimGrid = (seq_blocks, batch_size * head_nums);
    dim3 dimBlock = size(tiled_mma); // 128 线程

    static constexpr int smem_size = (cosize(sQ) + cosize(sKV) * 2) * sizeof(Element);

    auto kernel = flash_attn_cute_kernel<
        decltype(tensor_shape), decltype(cta_tiler), 
        decltype(sQ_layout), decltype(sKV_layout),
        decltype(tiled_mma0), decltype(tiled_mma1),
        decltype(copyQKV_g2s), decltype(copyQ_s2r), 
        decltype(copyK_s2r), decltype(copyV_s2r)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        query, key, value, output,
        scale, tensor_shape, cta_tiler,
        sQ_layout, sKV_layout,
        tiled_mma0, tiled_mma1,
        copyQKV_g2s, copyQ_s2r, copyK_s2r, copyV_s2r
    );
}

} // namespace fray