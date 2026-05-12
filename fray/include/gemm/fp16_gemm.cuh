#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace fray{

using namespace cute;

using Element = half;

template <typename ShapeMNK, typename CtaTiler, typename TiledMma,
          typename SmemLayoutA, typename TiledCopyA_g2s, typename TiledCopyA_s2r,
          typename SmemLayoutB, typename TiledCopyB_g2s, typename TiledCopyB_s2r,
          typename SmemLayoutC, typename TiledCopyC_s2g, typename TiledCopyC_r2s>
__global__ void fp16_gemm_cute_kernel(
    ShapeMNK shape_MNK, CtaTiler cta_tiler,
    Element const* Aptr, SmemLayoutA sA_layout, TiledCopyA_g2s copy_a_g2s, TiledCopyA_s2r copy_a_s2r,
    Element const* Bptr, SmemLayoutB sB_layout, TiledCopyB_g2s copy_b_g2s, TiledCopyB_s2r copy_b_s2r,
    Element*  Cptr, SmemLayoutC sC_layout, TiledCopyC_s2g copy_c_s2g, TiledCopyC_r2s copy_c_r2s,
    TiledMma tiled_mma) 
{
    // Global Tensor
    // A[M, K] , B[N, K] , C[M, N] 
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), select<0, 2>(shape_MNK), make_stride(select<2>(shape_MNK), _1{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), select<1, 2>(shape_MNK), make_stride(select<2>(shape_MNK), _1{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), select<0, 1>(shape_MNK), make_stride(select<1>(shape_MNK), _1{}));


    // CTA Tiling
    auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _); 
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k_tiles)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, k_tiles)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)


    // Shared Memory
    extern __shared__ char smem_raw[];
    Element* smemA_ptr = reinterpret_cast<Element*>(smem_raw);
    Element* smemB_ptr = smemA_ptr + cosize(sA_layout);
    
    Tensor sA = make_tensor(make_smem_ptr(smemA_ptr), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB_ptr), sB_layout); // (BLK_N, BLK_K, PIPE)


    // Global to Shared
    auto thr_copy_a = copy_a_g2s.get_thread_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, PIPE)

    auto thr_copy_b = copy_b_g2s.get_thread_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k_tiles)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, PIPE)


    // MMA & Shared to Register
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC);          // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    auto thr_copy_a_s2r = copy_a_s2r.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_copy_a_s2r.partition_S(sA);
    Tensor tCrA_view = thr_copy_a_s2r.retile_D(tCrA); // 视图重塑以匹配 CopyAtom

    auto thr_copy_b_s2r = copy_b_s2r.get_thread_slice(threadIdx.x);
    Tensor tCsB = thr_copy_b_s2r.partition_S(sB);
    Tensor tCrB_view = thr_copy_b_s2r.retile_D(tCrB);

    // Pipeline Setting
    int k_tile_count = size<3>(tAgA);
    int k_tile_next  = 0;
    const int K_PIPE_MAX = size<3>(tAsA);

    // Preload (K_PIPE_MAX -1) Tiles From Global To Shared
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_a_g2s, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b_g2s, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) ++k_tile_next;
    }

    // Init Read & Write Ptr
    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    // Prefetch the first thread_tile from shared to reg
    int REG_PIPELINE = size<2>(tCrA);
    if (REG_PIPELINE > 0) {
        cp_async_wait<K_PIPE_MAX - 2>(); // Ensure the first tile is in shared memory
        __syncthreads();
        copy(copy_a_s2r, tCsA(_, _, Int<0>{}, smem_pipe_read), tCrA_view(_, _, Int<0>{}));
        copy(copy_b_s2r, tCsB(_, _, Int<0>{}, smem_pipe_read), tCrB_view(_, _, Int<0>{}));
    }

    // Main Loop
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < REG_PIPELINE; ++k_block)
        {
            // Sync For Next Stage Tile
            if (k_block == REG_PIPELINE - 1)
            {
                cp_async_wait<K_PIPE_MAX - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + 1) % REG_PIPELINE;
            copy(copy_a_s2r, tCsA(_, _, k_block_next, smem_pipe_read), tCrA_view(_, _, k_block_next));
            copy(copy_b_s2r, tCsB(_, _, k_block_next, smem_pipe_read), tCrB_view(_, _, k_block_next));

            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copy_a_g2s, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                copy(copy_b_g2s, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0)
                {
                    ++k_tile_next;
                }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read + 1 == K_PIPE_MAX) ? 0 : smem_pipe_read + 1;
            }
            // Thread-level register gemm for k_block
            gemm(tiled_mma, tCrC, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // Write to share by using sA
    Tensor sC = make_tensor(sA.data(), sC_layout);

    auto thr_copy_c_r2s = copy_c_r2s.get_thread_slice(threadIdx.x);
    Tensor tCrC_r2s = thr_copy_c_r2s.retile_S(tCrC);  //[CPY, CPY_M, CPY_N]
    Tensor tCsC_r2s = thr_copy_c_r2s.partition_D(sC); //[CPY, CPY_M_sem, CPY_N_smem, PIPE_K = 4]

    auto thr_copy_c_s2g = copy_c_s2g.get_thread_slice(threadIdx.x);
    Tensor tCsC_s2g = thr_copy_c_s2g.partition_S(sC); //[CPY, CPY_M_smem, CPY_N_smem, PIPE]
    Tensor tCgC_s2g = thr_copy_c_s2g.partition_D(gC); //[CPY, CPY_M_gmem, CPY_N_gmem]

    // Descend multi dimension to linear
    Tensor tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // [CPY, TOTAL_TILES]
    Tensor tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // [CPY, TOTAL_TILES]

    int pipe_steps = size<3>(tCsC_r2s); // step = 4
    CUTE_UNROLL
    for (int i = 0; i < size<1>(tCrC_r2sx); i += pipe_steps) {
        // Reg -> Smem
        CUTE_UNROLL
        for (int j = 0; j < pipe_steps; ++j) {
            auto t = make_tensor_like<Element>(tCrC_r2sx(_, i + j));
            copy(tCrC_r2sx(_, i + j), t); 
            copy(copy_c_r2s, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

        // Smem -> Gmem (Coalesced)
        CUTE_UNROLL
        for (int j = 0; j < pipe_steps; ++j) {
            copy(copy_c_s2g, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

template <int BLOCK_M = 128, int BLOCK_N = 128, int NUM_STAGES = 3>
void fp16_gemm_c(int M, int N, int K,
                 Element const* A, Element const* B, Element* C,  
                 cudaStream_t stream = 0) 
{
    auto MNK_shape = make_shape(M, N, K);
    auto bM = Int<BLOCK_M>{};
    auto bN = Int<BLOCK_N>{};
    auto bK = Int<32>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<NUM_STAGES>{};

    // Shared Memory Layouts with Swizzling
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}))));
    
    auto sA_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bM, bK, bP));
    auto sB_layout = tile_to_shape(SmemLayoutAtom{}, make_shape(bN, bK, bP));

    // Epilogue Smem Layout
    auto sC_layout = tile_to_shape(make_layout(make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{})),
                                   make_shape(Int<32>{}, Int<32>{}, Int<4>{}));

    // Tiled MMA
    TiledMMA tiled_mma = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                        Layout<Shape<_2, _2, _1>>{},
                                        Tile<_32, _32, _16>{});

    // Copy Methods
    TiledCopy copy_a_g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                                           Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _8>>{}); // [Atom, Thread layout, Val layout]
    TiledCopy copy_b_g2s = copy_a_g2s;

    TiledCopy copy_a_s2r = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
    TiledCopy copy_b_s2r = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);

    TiledCopy copy_c_r2s = make_tiled_copy_C(Copy_Atom<UniversalCopy<int>, Element>{}, tiled_mma);
    TiledCopy copy_c_s2g = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                                           Layout<Shape<_32, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _8>>{});

    // Launch
    int smem_size = (cosize(sA_layout) + cosize(sB_layout)) * sizeof(Element);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));

    auto kernel = fp16_gemm_cute_kernel<
        decltype(MNK_shape), decltype(cta_tiler), decltype(tiled_mma),
        decltype(sA_layout), decltype(copy_a_g2s), decltype(copy_a_s2r),
        decltype(sB_layout), decltype(copy_b_g2s), decltype(copy_b_s2r),
        decltype(sC_layout), decltype(copy_c_s2g), decltype(copy_c_r2s)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, cta_tiler, 
        A, sA_layout, copy_a_g2s, copy_a_s2r,
        B, sB_layout, copy_b_g2s, copy_b_s2r,
        C, sC_layout, copy_c_s2g, copy_c_r2s,
        tiled_mma);
}

} // namespace fray