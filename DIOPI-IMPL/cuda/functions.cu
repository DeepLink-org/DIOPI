/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#include <diopi/functions.h>
#include <cuda_runtime.h>

#include "helper.hpp"


#define dispatch_dtype(fun, dtype, gridSize, blockSize, stream, ...)                             \
    if (diopi_dtype_int32 == dtype) {                                                            \
        fun<int32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    }                                                                                            \
    else if (diopi_dtype_uint32 == dtype) {                                                      \
        fun<uint32_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    }                                                                                            \
    else if (diopi_dtype_int16 == dtype) {                                                       \
        fun<int16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    }                                                                                            \
    else if (diopi_dtype_uint16 == dtype) {                                                      \
        fun<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                          \
    }                                                                                            \
    else if (diopi_dtype_int8 == dtype) {                                                        \
        fun<int8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    }                                                                                            \
    else if (diopi_dtype_uint8 == dtype) {                                                       \
        fun<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                           \
    }                                                                                            \
    else if (diopi_dtype_float32 == dtype) {                                                     \
        fun<float><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                             \
    }                                                                                            \
    else if (diopi_dtype_float64 == dtype) {                                                     \
        fun<double><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                            \
    }                                                                                            \
    else if (diopi_dtype_bool == dtype) {                                                        \
        fun<bool><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__);                              \
    }                                                                                            \
    else {                                                                                       \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, \
                dtype, gridSize, blockSize, #__VA_ARGS__);                                       \
        return diopiDtypeNotSupported;                                                           \
    }


template<typename T> __global__
void vecAdd(const void* a, const void* b, void* c, const int numel)
{
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A  = static_cast<const T*>(a);
    const T* B  = static_cast<const T*>(b);
    T*       C  = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + B[id];
    }
}

extern "C" diopiError_t add(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other)
{
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOther = impl::cuda::makeTensor(other);
    auto trOut   = impl::cuda::makeTensor(out);

    int blockSize = 256;
    int gridSize  = (trInput.numel() + blockSize - 1) / blockSize;
    dispatch_dtype(vecAdd, trInput.dtype(), gridSize, blockSize, stream,
        trInput.data(), trOther.data(), trOut.data(), trInput.numel());
    return diopiSuccess;
}


template<typename T> __global__
void vecFill(void* a, const T value, const int numel)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    T*  A  = static_cast<T*>(a);
    if (id < numel) {
        A[id] = value;
    }
}

extern "C" diopiError_t fill(diopiContextHandle_t ctx, diopiTensorHandle_t tensor, const float value)
{
    auto stream = impl::cuda::getStream(ctx);
    auto tr = impl::cuda::makeTensor(tensor);

    diopiDevice_t device = tr.device();
    diopiDtype_t  dtype  = tr.dtype();
    int64_t       numel  = tr.numel();

    if (diopi_host == device) {
        return diopiErrorOccurred;
    } else {
        int blockSize = 256;
        int gridSize  = (numel + blockSize - 1) / blockSize;
        dispatch_dtype(vecFill, dtype, gridSize, blockSize, stream, tr.data(), value, numel);
    }

    return diopiSuccess;
}
