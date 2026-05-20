#pragma once

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>


namespace fray {
namespace utils {

using namespace cute;

// Precision Conversion
template <typename To, typename Engine, typename Layout>
__device__ __forceinline__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To, From, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From, numel> *>(tensor.data())); // frag:cutlass::Array<To, numel>
    return make_tensor(make_rmem_ptr<To>(&frag), tensor.layout());
}


// Layout Transformations

// Convert from (MMA=4, MMA_M, MMA_N) to (MMA=8, MMA_M, MMA_K = MMA_N/2)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_c2a(Layout layout) {
    auto l = logical_divide(layout, Shape<X, X, _2>{}); // l:(_4, MMA_M, (2, MMA_N/2))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

// Convert from (MMA=8, MMA_M, MMA_K) to (MMA=4, MMA_M, MMA_N = MMA_K * 2)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_a2c(Layout layout) {
    auto l = logical_divide(layout, Shape<_2, X, X>{}); // l:((2, 4), MMA_M, MMA_N)
    return make_layout(get<0, 1>(l), get<1>(l), make_layout(get<0, 0>(l), get<2>(l)));
}

// Convert from (MMA=4, MMA_N, MMA_K) to (MMA=8, MMA_M = MMA_N/2, MMA_K)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_b2a(Layout layout) {
    auto l = logical_divide(layout, Shape<X, _2, X>{}); 
    return make_layout(make_layout(get<0>(l), get<1, 0>(l)), get<1, 1>(l), get<2>(l));
}

// Convert from (MMA=8, MMA_M, MMA_K) to (MMA=4, MMA_N = MMA_M*2, MMA_K)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_a2b(Layout layout) {
    auto l = logical_divide(layout, Shape<_2, X, X>{});
    return make_layout(get<0, 1>(l), make_layout(get<0, 0>(l), get<1>(l)), get<2>(l));
}

// Convert from (MMA=4, MMA_M, MMA_N) to (MMA=4, MMA_N, MMA_K = MMA_M)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_c2b(Layout layout) {
    return make_layout(get<0>(layout), get<2>(layout), get<1>(layout));
}

// Convert from (MMA=4, MMA_N, MMA_K) to (MMA=4, MMA_M = MMA_K, MMA_N)
template <typename Layout>
__device__ __forceinline__ auto convert_reg_layout_b2c(Layout layout) {
    return make_layout(get<0>(layout), get<2>(layout), get<1>(layout));
}

// Logical View Transformations

// Convert from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
__device__ __forceinline__ auto mma_to_rowcol_layout(Layout acc_layout){
    auto l = logical_divide(acc_layout, Shape<_2>{}); // l: ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), // 逻辑行
                       make_layout(get<0, 0>(l), get<2>(l)));// 逻辑列
}

} // namespace utils
} // namespace fray