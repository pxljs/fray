#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>

namespace fray {
namespace topk_detail {
// Valid entries precede padding. NaNs sort first for largest, last for smallest.
// Equal values (including signed zero and NaNs) use the lower source index.
template <bool Largest>
__device__ __forceinline__ bool better(float a, int ai, float b, int bi) {
    if (ai == INT_MAX) return false;
    if (bi == INT_MAX) return true;
    const bool an = isnan(a), bn = isnan(b);
    if (an != bn) return Largest ? an : !an;
    if ((an && bn) || a == b) return ai < bi;
    return Largest ? a > b : a < b;
}

template <bool Largest>
__device__ __forceinline__ void warp_best(float& v, int& i) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, offset);
        int oi = __shfl_xor_sync(0xffffffff, i, offset);
        if (better<Largest>(other, oi, v, i)) { v = other; i = oi; }
    }
}

// Groups of 32 threads handle short rows; 256 threads cooperate on wide rows.
// The input is read once. Selection repeatedly reduces register-local winners.
template <typename T, int Items, int Threads, bool Largest>
__global__ void kernel(const T* __restrict__ x, T* __restrict__ values,
                       int64_t* __restrict__ indices, int rows, int n, int k) {
    constexpr bool WarpRow = Threads == 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = WarpRow ? lane : threadIdx.x;
    const int row = WarpRow ? blockIdx.x * 4 + warp : blockIdx.x;
    if (row >= rows) return; // Uniform for each warp (or entire block).
    float data[Items];
    int ids[Items];
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        int col = tid + j * Threads;
        ids[j] = col < n ? col : INT_MAX;
        data[j] = col < n ? float(x[int64_t(row) * n + col]) : 0.f;
    }
    __shared__ float warp_values[8];
    __shared__ int warp_indices[8];
    for (int rank = 0; rank < k; ++rank) {
        float best = 0.f;
        int index = INT_MAX;
        #pragma unroll
        for (int j = 0; j < Items; ++j) {
            if (better<Largest>(data[j], ids[j], best, index)) {
                best = data[j]; index = ids[j];
            }
        }
        warp_best<Largest>(best, index);
        if constexpr (!WarpRow) {
            if (lane == 0) { warp_values[warp] = best; warp_indices[warp] = index; }
            __syncthreads();
            if (warp == 0) {
                best = lane < 8 ? warp_values[lane] : 0.f;
                index = lane < 8 ? warp_indices[lane] : INT_MAX;
                warp_best<Largest>(best, index);
                if (lane == 0) { warp_values[0] = best; warp_indices[0] = index; }
            }
            __syncthreads();
            index = warp_indices[0];
            // All threads finish reading shared storage before the next rank.
            __syncthreads();
        }
        if (tid == 0) {
            values[int64_t(row) * k + rank] = x[int64_t(row) * n + index];
            indices[int64_t(row) * k + rank] = index;
        }
        #pragma unroll
        for (int j = 0; j < Items; ++j) {
            if (ids[j] == index) ids[j] = INT_MAX;
        }
    }
}

// Ascending unsigned keys encode the requested value order, then source index.
// Canonicalizing NaNs/zeros affects comparison only; output reads original bits.
template <bool Largest>
__device__ __forceinline__ uint64_t sort_key(float v, uint32_t index) {
    uint32_t bits = __float_as_uint(v == 0.f ? 0.f : v);
    uint32_t ordered = isnan(v) ? 0xffffffffu
        : ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u));
    if constexpr (Largest) ordered = ~ordered;
    return (uint64_t(ordered) << 16) | index;
}

// A maximum of 4096 keys per block keeps CUB exchange storage below 48 KiB.
// Keys contain indices, so the initial per-thread arrangement need not be
// blocked: loads and SortBlockedToStriped output stores are both coalesced.
template <typename T, int Items, bool Largest, bool Tiled>
__global__ void radix_kernel(const T* __restrict__ x, T* __restrict__ values,
                             int64_t* __restrict__ indices,
                             uint64_t* __restrict__ workspace, int n, int k,
                             int tiles) {
    constexpr int Tile = 256 * Items;
    const int row = blockIdx.x / tiles;
    const int tile = blockIdx.x % tiles;
    using Sort = cub::BlockRadixSort<uint64_t, 256, Items>;
    __shared__ typename Sort::TempStorage temp;
    uint64_t keys[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int col = tile * Tile + threadIdx.x + j * 256;
        keys[j] = col < n ? sort_key<Largest>(float(x[int64_t(row) * n + col]), col)
                          : UINT64_MAX;
    }
    // 32 value bits + 16 index bits; upper 16 bits need no radix passes.
    Sort(temp).SortBlockedToStriped(keys, 0, 48);
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int rank = threadIdx.x + j * 256;
        if constexpr (Tiled) {
            workspace[(int64_t(row) * tiles + tile) * Tile + rank] = keys[j];
        } else if (rank < k) {
            const uint32_t index = uint32_t(keys[j] & 0xffffu);
            values[int64_t(row) * k + rank] = x[int64_t(row) * n + index];
            indices[int64_t(row) * k + rank] = index;
        }
    }
}

// Merge up to four sorted runs by finding each key's rank in the other runs.
// Composite keys are unique for real elements, so final stores cannot race.
// A local rank >= k cannot enter the global top-k and requires no search.
__global__ void merge_radix_kernel(const uint64_t* __restrict__ workspace,
                                   const void* input, void* output,
                                   int64_t* __restrict__ indices,
                                   int n, int k, int tiles, int element_bytes) {
    constexpr int Tile = 4096;
    const int row = blockIdx.x / tiles;
    const int tile = blockIdx.x % tiles;
    const uint64_t* runs = workspace + int64_t(row) * tiles * Tile;
    for (int local = threadIdx.x; local < min(k, Tile); local += blockDim.x) {
        const uint64_t key = runs[tile * Tile + local];
        if (key == UINT64_MAX) continue;
        int rank = local;
        for (int other = 0; other < tiles; ++other) {
            if (other == tile) continue;
            int lo = 0, hi = min(Tile, n - other * Tile);
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (runs[other * Tile + mid] < key) lo = mid + 1;
                else hi = mid;
            }
            rank += lo;
        }
        if (rank < k) {
            const uint32_t index = uint32_t(key & 0xffffu);
            const int64_t dst = int64_t(row) * k + rank;
            const int64_t src = int64_t(row) * n + index;
            // Bitwise copies retain signed zeros and NaN payloads.
            if (element_bytes == 4)
                static_cast<uint32_t*>(output)[dst] = static_cast<const uint32_t*>(input)[src];
            else
                static_cast<uint16_t*>(output)[dst] = static_cast<const uint16_t*>(input)[src];
            indices[dst] = index;
        }
    }
}

} // namespace topk_detail

// Contiguous [rows, n], disjoint output buffers [rows, k]. Caller owns storage.
// Items * Threads must cover n. No allocation, synchronization or default-stream
// dependency. CUDA graph compatible once the caller has compiled the kernel.
template <typename T, int Items, int Threads, bool Largest = true>
cudaError_t topk_c(const T* x, T* values, int64_t* indices,
                   int rows, int n, int k, cudaStream_t stream = nullptr) {
    static_assert(Threads == 32 || Threads == 256, "Unsupported row group");
    static_assert(Items > 0 && Items <= 64, "Invalid register tile");
    if (rows < 0 || n < 0 || k < 0 || k > n || n > Items * Threads)
        return cudaErrorInvalidValue;
    if (!rows || !k) return cudaSuccess;
    if (!x || !values || !indices) return cudaErrorInvalidValue;
    const unsigned blocks = Threads == 32 ? (unsigned(rows) + 3) / 4 : unsigned(rows);
    topk_detail::kernel<T, Items, Threads, Largest>
        <<<blocks, Threads == 32 ? 128 : 256, 0, stream>>>(x, values, indices, rows, n, k);
    return cudaGetLastError();
}

// For n > 4096, workspace must hold rows * ceil(n / 4096) * 4096 uint64 keys.
// For smaller rows workspace is unused. Items must cover the row (small case)
// or equal 16 (tiled case). Caller owns the stream-ordered workspace lifetime.
template <typename T, int Items, bool Largest = true>
cudaError_t topk_radix_c(const T* x, T* values, int64_t* indices,
                         uint64_t* workspace, int rows, int n, int k,
                         cudaStream_t stream = nullptr) {
    static_assert(Items > 0 && Items <= 16, "Invalid radix tile");
    if (rows < 0 || n < 0 || n > 16384 || k < 0 || k > n)
        return cudaErrorInvalidValue;
    if (!rows || !k) return cudaSuccess;
    if (!x || !values || !indices) return cudaErrorInvalidValue;
    if (n <= 4096) {
        if (n > Items * 256) return cudaErrorInvalidValue;
        topk_detail::radix_kernel<T, Items, Largest, false>
            <<<rows, 256, 0, stream>>>(x, values, indices, workspace, n, k, 1);
        return cudaGetLastError();
    }
    if constexpr (Items == 16) {
        const int tiles = (n + 4095) / 4096;
        const uint64_t blocks = uint64_t(rows) * tiles;
        if (!workspace || blocks > INT_MAX) return cudaErrorInvalidValue;
        topk_detail::radix_kernel<T, 16, Largest, true>
            <<<unsigned(blocks), 256, 0, stream>>>(x, values, indices, workspace, n, k, tiles);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return error;
        topk_detail::merge_radix_kernel<<<unsigned(blocks), 256, 0, stream>>>(
            workspace, x, values, indices, n, k, tiles, sizeof(T));
        return cudaGetLastError();
    }
    return cudaErrorInvalidValue;
}

} // namespace fray
