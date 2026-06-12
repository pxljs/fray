#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include "flash_mla/softmax.cuh"
#include "utils.cuh"

// #define CUTE_DEBUG

namespace fray {
    using namespace cute;
    using namespace fray::utils;
    using Element = cute::half_t;

template <
    int BlockM = 16,   // 每个 Block 处理的 Query 头数
    int BlockN = 64,   // 每个 Block 迭代的序列分块长度
    int DimNope = 512, // 隐空间通道长度
    int DimRope = 64,  // 旋转编码通道长度
    int BlockK = 128   // 隐空间的 Tiling 宽度（512 分 4 次加载）
>
struct FlashMLATraits {
    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = BlockN;
    static constexpr int kDimNope = DimNope;
    static constexpr int kDimRope = DimRope;
    static constexpr int kBlockK = BlockK;
    static constexpr int kDim = DimNope + DimRope;

    // using Swizzle8B = Swizzle<3, 3, 3>;
    // using SmemLayoutQ_Nope = decltype(composition(
    //     Swizzle8B{},
    //     make_layout(make_shape(Int<kBlockM>{}, Int<kDimNope>{}), 
    //                 make_stride(Int<kDimNope>{}, _1{}))
    // ));
    // using SmemLayoutK_Nope = decltype(composition(
    //     Swizzle8B{},
    //     make_layout(make_shape(Int<kBlockN>{}, Int<kDimNope>{}), 
    //                 make_stride(Int<kDimNope>{}, _1{}))
    // ));
    // using SmemLayoutQ_Rope = decltype(composition(
    //     Swizzle8B{},
    //     make_layout(make_shape(Int<kBlockM>{}, Int<kDimRope>{}), 
    //                 make_stride(Int<kDimRope>{}, _1{}))
    // ));
    // using SmemLayoutK_Rope = decltype(composition(
    //     Swizzle8B{},
    //     make_layout(make_shape(Int<kBlockN>{}, Int<kDimRope>{}), 
    //                 make_stride(Int<kDimRope>{}, _1{}))
    // ));

    using SmemLayoutQ_Nope = Layout<Shape<Int<kBlockM>, Int<kDimNope>>, Stride<Int<kDimNope>, _1>>;
    using SmemLayoutK_Nope = Layout<Shape<Int<kBlockN>, Int<kDimNope>>, Stride<Int<kDimNope>, _1>>;

    using SmemLayoutQ_Rope = Layout<Shape<Int<kBlockM>, Int<kDimRope>>, Stride<Int<kDimRope>, _1>>;
    using SmemLayoutK_Rope = Layout<Shape<Int<kBlockN>, Int<kDimRope>>, Stride<Int<kDimRope>, _1>>;

    using SmemLayoutV_t = decltype(composition(SmemLayoutK_Nope{}, make_layout(make_shape(Int<kDimNope>{}, Int<kBlockN>{}), GenRowMajor{})));

    using TiledMma = decltype(make_tiled_mma(
        SM80_16x8x16_F32F16F16F32_TN{},
        Layout<Shape<_1, _1, _1>>{}));
};

template <typename Traits>
__global__ void flash_mla_paged_decoding_kernel(
    const Element* __restrict__ Q,          // [Batch, 1, H_q, 576] （前 512 维为 absorbed Q_nope，后 64 维为 Q_rope）
    const Element* __restrict__ KV_cache,   // 分页显存池 [Max_Pages, Block_Size=64, 576] （前 512 维为 c_KV，后 64 维为 k_RoPE）
    const int* __restrict__ block_table,    // 分页映射表 [Batch, Max_Blocks_Per_Seq]
    const int* __restrict__ seq_lens,       // 每个 Batch 的实际有效序列长度 [Batch]
    Element* __restrict__ O_intermediate,   // [Batch, H_q, 512] 
    int H_q,
    int max_blocks_per_seq,
    float scale)
{
    // 一个block处理一个序列的Q的一部分head
    int b = blockIdx.x;
    int h_group = blockIdx.y;
    int h_start = h_group * Traits::kBlockM;
    if(h_start >= H_q) return; // 超出实际头数范围的 Block 直接退出

    int seq_len = seq_lens[b];
    if (seq_len == 0) return;
    int thread_id = threadIdx.x;
    const Element* q_base_ptr = Q + (int64_t)b * H_q * Traits::kDim + h_start * Traits::kDim;
    Element* o_base_ptr = O_intermediate + (int64_t)b * H_q * Traits::kDimNope + h_start * Traits::kDimNope;

#ifdef CUTE_DEBUG
    if (thread0()) {
        printf("\n--- Kernel Start [b=%d, h_group=%d] ---\n", b, h_group);
    }
#endif

    // Global tensor
    Tensor gQ_nope = make_tensor(make_gmem_ptr(q_base_ptr), 
                                 make_shape(Int<Traits::kBlockM>{}, Int<Traits::kDimNope>{}), 
                                 make_stride(Int<Traits::kDim>{}, _1{}));
    Tensor gQ_rope = make_tensor(make_gmem_ptr(q_base_ptr + Traits::kDimNope), 
                                 make_shape(Int<Traits::kBlockM>{}, Int<Traits::kDimRope>{}), 
                                 make_stride(Int<Traits::kDim>{}, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(o_base_ptr), 
                                 make_shape(Int<Traits::kBlockM>{}, Int<Traits::kDimNope>{}), 
                                 make_stride(Int<Traits::kDimNope>{}, _1{}));                    


    extern __shared__ char smem_raw[];
    Element* sq_nope_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* sk_nope_ptr = sq_nope_ptr + Traits::kBlockM * Traits::kDimNope;
    Element* sq_rope_ptr = sk_nope_ptr + Traits::kBlockN * Traits::kDimNope;
    Element* sk_rope_ptr = sq_rope_ptr + Traits::kBlockM * Traits::kDimRope;

    auto sQ_rope = make_tensor(make_smem_ptr(sq_rope_ptr), typename Traits::SmemLayoutQ_Rope{});
    auto sK_rope = make_tensor(make_smem_ptr(sk_rope_ptr), typename Traits::SmemLayoutK_Rope{});
    auto sQ_nope = make_tensor(make_smem_ptr(sq_nope_ptr), typename Traits::SmemLayoutQ_Nope{});
    auto sK_nope = make_tensor(make_smem_ptr(sk_nope_ptr), typename Traits::SmemLayoutK_Nope{});
    auto sVt = make_tensor(make_smem_ptr(sk_nope_ptr), typename Traits::SmemLayoutV_t{});

    auto sQ_nope_tiles = local_tile(sQ_nope, make_tile(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}), make_coord(_0{}, _));
    auto sK_nope_tiles = local_tile(sK_nope, make_tile(Int<Traits::kBlockN>{}, Int<Traits::kBlockK>{}), make_coord(_0{}, _));
    auto sVt_tiles = local_tile(sVt, make_tile(Int<Traits::kBlockK>{}, Int<Traits::kBlockN>{}), make_coord(_, _0{}));

    // 申请 O_acc 寄存器（16 x 512，分为 4 个 128 维分块以节省寄存器并对应 MMA）
    typename Traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_id);
    Tensor tOrO_0 = partition_fragment_C(tiled_mma, make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}));
    Tensor tOrO_1 = partition_fragment_C(tiled_mma, make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}));
    Tensor tOrO_2 = partition_fragment_C(tiled_mma, make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}));
    Tensor tOrO_3 = partition_fragment_C(tiled_mma, make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}));
    clear(tOrO_0);
    clear(tOrO_1); 
    clear(tOrO_2); 
    clear(tOrO_3);

    // 预加载当前 Block 负责的 Q_rope (16 x 64) 并在共享内存中持久化
    using CopyThreadLayout = Layout<Shape<Int<Traits::kBlockM>, Int<32 / Traits::kBlockM>>, 
                                    Stride<Int<32 / Traits::kBlockM>, _1>>;
    auto copy_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                                    CopyThreadLayout{}, Layout<Shape<_1, _8>>{});
    auto thr_copy_g2s = copy_g2s.get_thread_slice(thread_id);
    Tensor tQgQ_nope = thr_copy_g2s.partition_S(gQ_nope);
    Tensor tQsQ_nope = thr_copy_g2s.partition_D(sQ_nope);
    copy(copy_g2s, tQgQ_nope, tQsQ_nope);

    Tensor tQgQ_rope = thr_copy_g2s.partition_S(gQ_rope);
    Tensor tQsQ_rope = thr_copy_g2s.partition_D(sQ_rope);
    copy(copy_g2s, tQgQ_rope, tQsQ_rope);

    cp_async_fence();
    __syncthreads();

    Tensor tSrS = partition_fragment_C(tiled_mma, make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockN>{}));
    Tensor cS = make_identity_tensor(make_shape(Int<Traits::kBlockM>{}, Int<Traits::kBlockN>{}));
    Tensor tScS = thr_mma.partition_C(cS);
    static constexpr int kRowsPerThread = decltype(size<0>(mma_to_rowcol_layout(tSrS.layout())))::value;
    
    // 初始化变参 OnlineSoftmax
    OnlineSoftmax<kRowsPerThread> softmax;
    float scale_log2 = scale * 1.44269504f; // scale * log2(e)

    // 3. 主循环：沿 Token 序列长度以 Block_N = 64 为步长迭代
    int num_seq_tiles = (seq_len + Traits::kBlockN - 1) / Traits::kBlockN;
    for (int tile_idx = 0; tile_idx < num_seq_tiles; ++tile_idx) {
        int token_start = tile_idx * Traits::kBlockN;

        // 页表映射计算：获取当前分块（Page）在分页显存池中的基地址
        int page_idx = block_table[b * max_blocks_per_seq + tile_idx];
        const Element* page_base_ptr = KV_cache + (int64_t)page_idx * Traits::kBlockN * Traits::kDim;

        Tensor gK_nope = make_tensor(make_gmem_ptr(page_base_ptr), 
                                    make_shape(Int<Traits::kBlockN>{}, Int<Traits::kDimNope>{}), 
                                    make_stride(Int<Traits::kDim>{}, _1{}));
        Tensor gK_rope = make_tensor(make_gmem_ptr(page_base_ptr + Traits::kDimNope), 
                                    make_shape(Int<Traits::kBlockN>{}, Int<Traits::kDimRope>{}), 
                                    make_stride(Int<Traits::kDim>{}, _1{}));

        Tensor tKgK_rope = thr_copy_g2s.partition_S(gK_rope);
        Tensor tKsK_rope = thr_copy_g2s.partition_D(sK_rope);
        copy(copy_g2s, tKgK_rope, tKsK_rope);

        cp_async_fence();

        // 计算 S_rope
        clear(tSrS);
        // 必须等待Q、K已经搬运到共享内存
        cp_async_wait<0>();
        __syncthreads();
        Tensor tSsQ_rope = thr_mma.partition_A(sQ_rope);
        Tensor tSsK_rope = thr_mma.partition_B(sK_rope);
        Tensor tSrQ_rope = thr_mma.partition_fragment_A(sQ_rope);
        Tensor tSrK_rope = thr_mma.partition_fragment_B(sK_rope);
        copy(tSsQ_rope, tSrQ_rope);
        copy(tSsK_rope, tSrK_rope);

        // 计算前发送异步搬运k_nope指令
        Tensor tKgK_nope = thr_copy_g2s.partition_S(gK_nope);
        Tensor tKsK_nope = thr_copy_g2s.partition_D(sK_nope);
        copy(copy_g2s, tKgK_nope, tKsK_nope);
        cp_async_fence();

        gemm(tiled_mma, tSrS, tSrQ_rope, tSrK_rope, tSrS);


        // Cal nope scores
        cp_async_wait<0>();
        __syncthreads();

        CUTE_UNROLL
        for (int c = 0; c < 4; ++c) {
            auto sQ_nope_chunk = sQ_nope_tiles(_, _, c);
            auto sK_nope_chunk = sK_nope_tiles(_, _, c);

            // 计算 S_nope 分块并累加
            Tensor tSsQ_nope = thr_mma.partition_A(sQ_nope_chunk);
            Tensor tSsK_nope = thr_mma.partition_B(sK_nope_chunk);
            Tensor tSrQ_nope = thr_mma.partition_fragment_A(sQ_nope_chunk);
            Tensor tSrK_nope = thr_mma.partition_fragment_B(sK_nope_chunk);
            copy(tSsQ_nope, tSrQ_nope);
            copy(tSsK_nope, tSrK_nope);
            gemm(tiled_mma, tSrS, tSrQ_nope, tSrK_nope, tSrS);
        }

        int valid_tokens = seq_len - token_start;
        if (valid_tokens < Traits::kBlockN) {
            CUTE_UNROLL
            for (int i = 0; i < size(tSrS); ++i) {
                int token_offset = int(get<1>(tScS(i)));
                if (token_offset >= valid_tokens) {
                    tSrS(i) = -INFINITY;
                }
            }
        }

#ifdef CUTE_DEBUG
        if (thread0()) {
            printf("Tile %d: before softmax, tSrS(0) = %f\n", tile_idx, float(tSrS(0)));
        }
#endif

        // Online-Softmax
        softmax.update(tSrS, scale_log2, tOrO_0, tOrO_1, tOrO_2, tOrO_3);

        // 寄存器级 FP32 至 FP16 精度压缩
        Tensor rP = convert_type<Element>(tSrS);
        Tensor tOrP = make_tensor(rP.data(), convert_reg_layout_c2a(rP.layout()));

        auto update_o_chunk = [&](auto& accum_o, auto c) {
            auto sVt_chunk = sVt_tiles(_, _, c);
            Tensor tOsVt = thr_mma.partition_B(sVt_chunk);
            Tensor tOrVt = thr_mma.partition_fragment_B(sVt_chunk);
            copy(tOsVt, tOrVt);

            gemm(tiled_mma, accum_o, tOrP, tOrVt, accum_o);
        };

        // O = P * V
        update_o_chunk(tOrO_0, Int<0>{});
        update_o_chunk(tOrO_1, Int<1>{});
        update_o_chunk(tOrO_2, Int<2>{});
        update_o_chunk(tOrO_3, Int<3>{});

#ifdef CUTE_DEBUG
        if (thread0()) {
            printf("Tile %d: after gemm, tOrO_0(0) = %f\n", tile_idx, float(tOrO_0(0)));
        }
#endif

    }

    // Finalize
    softmax.finalize(tOrO_0, tOrO_1, tOrO_2, tOrO_3);

#ifdef CUTE_DEBUG
        if (thread0()) {
            printf(
            "O0=%f O1=%f O2=%f O3=%f\n",
            float(tOrO_0(0)),
            float(tOrO_1(0)),
            float(tOrO_2(0)),
            float(tOrO_3(0))
            );
    }
#endif

    auto copyO_r2s = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, Element>{}, tiled_mma);
    auto thr_o_r2s = copyO_r2s.get_thread_slice(thread_id);
    auto sO = make_tensor(make_smem_ptr(sq_nope_ptr), typename Traits::SmemLayoutQ_Nope{});
    auto sO_tiles = local_tile(sO, make_tile(Int<Traits::kBlockM>{}, Int<Traits::kBlockK>{}), make_coord(_0{}, _));

    auto write_o_chunk = [&](auto const& accum_o, auto c) {
        Tensor rO = convert_type<Element>(accum_o);

#ifdef CUTE_DEBUG
        // 打印 1: 检查 FP32 转 FP16 后的寄存器数值
        if (thread0() && b == 0 && h_group == 0) {
            printf("\n[WriteBack] --- Block C = %d ---\n", int(c));
            printf("[WriteBack] 1. Reg rO(0) = %f, rO(1) = %f\n", float(rO(0)), float(rO(1)));
        }
#endif

        // 寄存器写回共享内存
        auto sO_chunk = sO_tiles(_, _, c);
        auto tOsO_r2s = thr_o_r2s.partition_D(sO_chunk);
        auto tOrO_r2s = thr_o_r2s.retile_S(rO);
        copy(copyO_r2s, tOrO_r2s, tOsO_r2s);
    };

    write_o_chunk(tOrO_0, Int<0>{});
    write_o_chunk(tOrO_1, Int<1>{});
    write_o_chunk(tOrO_2, Int<2>{});
    write_o_chunk(tOrO_3, Int<3>{});
    __syncthreads();

#ifdef CUTE_DEBUG
        // 打印 2: 检查 CuTe 的 copy 是否成功将数据写入 Shared Memory
        if (thread0() && b == 0 && h_group == 0) {
            printf("[WriteBack] 2. SMEM sq_nope_ptr[0] = %f, sq_nope_ptr[1] = %f\n", 
                   float(sq_nope_ptr[0]), float(sq_nope_ptr[1]));
        }
#endif

    auto copy_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{}, 
                                    CopyThreadLayout{}, Layout<Shape<_1, _8>>{});
    auto thr_copy_s2g = copy_s2g.get_thread_slice(thread_id);

    Tensor tOsO = thr_copy_s2g.partition_S(sO);
    Tensor tOgO = thr_copy_s2g.partition_D(gO);
    copy(copy_s2g, tOsO, tOgO);


#ifdef CUTE_DEBUG
        // 打印 3: 检查最终的全局显存地址处是否有值
        if (thread0() && b == 0 && h_group == 0) {
            printf("[WriteBack] 3. GMEM o_base_ptr[0] = %f\n", float(o_base_ptr[0]));
        }
#endif
}


template <int R = 16, int C = 64, int D_nope = 512, int D_rope = 64, int K = 128>
void flash_mla_paged_decoding(
    const Element* Q,          // [Batch, 1, H_q, 576]
    const Element* KV_cache,   // [Max_Pages, Block_Size=64, 576]
    const int* block_table,    // [Batch, Max_Blocks_Per_Seq]
    const int* seq_lens,       // [Batch]
    Element* O_intermediate,   // [Batch, H_q, 512]
    int batch_size,
    int num_heads,
    int max_blocks_per_seq,
    float scale,
    cudaStream_t stream) 
{
    using Traits = FlashMLATraits<R, C, D_nope, D_rope, K>;

    int grid_y = (num_heads + R - 1) / R;
    dim3 grid(batch_size, grid_y);
    dim3 block(32);

    static constexpr int smem_size = (R + C) * Traits::kDim * sizeof(Element);

    auto kernel = flash_mla_paged_decoding_kernel<Traits>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<grid, block, smem_size, stream>>>(
        Q, KV_cache, block_table, seq_lens, O_intermediate,
        num_heads, max_blocks_per_seq, scale
    );
    
}

} // namespace fray