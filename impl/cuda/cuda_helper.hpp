/**
 * @file cuda_helper.hpp
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CUDA_CUDA_HELPER_HPP_
#define IMPL_CUDA_CUDA_HELPER_HPP_

#include <cuda.h>

#include <algorithm>

#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)            \
    for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
        for (size_t j = blockIdx.y; j < (m); j += gridDim.y)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
    int optimal_block_num = (N + num_threads - 1) / num_threads;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height, const int width, T y, T x, const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = input[y_low * width + x_low];
    T v2 = input[y_low * width + x_high];
    T v3 = input[y_high * width + x_low];
    T v4 = input[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4, int& x_low, int& x_high, int& y_low,
                                              int& y_high, const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = static_cast<int>(y);
    x_low = static_cast<int>(x);

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;

    // reference in forward
    // T v1 = input[y_low * width + x_low];
    // T v2 = input[y_low * width + x_high];
    // T v3 = input[y_high * width + x_low];
    // T v4 = input[y_high * width + x_high];
    // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return;
}

#define DISPATCH_DTYPE(fun, dtype, gridSize, blockSize, stream, ...)                                                                        \
    if (diopi_dtype_int32 == dtype) {                                                                                                       \
        fun<int32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                      \
    } else if (diopi_dtype_uint32 == dtype) {                                                                                               \
        fun<uint32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                     \
    } else if (diopi_dtype_int16 == dtype) {                                                                                                \
        fun<int16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                      \
    } else if (diopi_dtype_uint16 == dtype) {                                                                                               \
        fun<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                     \
    } else if (diopi_dtype_int8 == dtype) {                                                                                                 \
        fun<int8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                       \
    } else if (diopi_dtype_uint8 == dtype) {                                                                                                \
        fun<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                      \
    } else if (diopi_dtype_float32 == dtype) {                                                                                              \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                        \
    } else if (diopi_dtype_float64 == dtype) {                                                                                              \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                       \
    } else if (diopi_dtype_bool == dtype) {                                                                                                 \
        fun<bool><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                         \
    } else {                                                                                                                                \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, dtype, gridSize, blockSize, #__VA_ARGS__); \
        return diopiDtypeNotSupported;                                                                                                      \
    }

#define DISPATCH_FLOAT_TYPES(fun, dtype, gridSize, blockSize, stream, ...)                                                                  \
    if (diopi_dtype_float32 == dtype) {                                                                                                     \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                        \
    } else if (diopi_dtype_float64 == dtype) {                                                                                              \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                                                                       \
    } else {                                                                                                                                \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, dtype, gridSize, blockSize, #__VA_ARGS__); \
        return diopiDtypeNotSupported;                                                                                                      \
    }
#endif  // IMPL_CUDA_CUDA_HELPER_HPP_
