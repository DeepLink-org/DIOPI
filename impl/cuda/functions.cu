/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

#include "helper.hpp"
#include "cuda_helper.hpp"

template<typename T> __global__
void vecAdd(const void* a, const void* b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * B[id];
    }
}

template<typename T> __global__
void vecAddBroadcast(const void* a, const void* b, void* c, const int numel, const T alpha,
        const int64_t* stride1, const int64_t* stride2, const int64_t* outStride, const int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    const T* B = static_cast<const T*>(b);
    T* C = static_cast<T*>(c);
    int size = id;
    size_t idxA = 0;
    size_t idxB = 0;
    if (id < numel) {
        for (int i = 0; i < len; ++i) {
            int tmp = size / outStride[i];
            idxA += tmp * stride1[i];
            idxB += tmp * stride2[i];
            size = size % outStride[i];
        }
        C[id] = A[idxA] + alpha * B[idxB];
    }
}

template<typename T> __global__
void vecAddScalar(const void* a, const T b, void* c, const int numel, const T alpha) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A = static_cast<const T*>(a);
    T* C = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + alpha * b;
    }
}

bool compareShape(const diopiSize_t& size1, const diopiSize_t& size2) {
    if (size1.len == size2.len) {
        for (int i = 0; i < size1.len; ++i) {
            if (size1.data[i] != size2.data[i]) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

void computeStride(const diopiSize_t& size1, const diopiSize_t& size2, diopiSize_t outSize,
        int64_t* stride1, int64_t* stride2) {
    int length = size1.len;
    int len = outSize.len;
    int64_t stride = 1;
    for (int i = 0; i < len; ++i) {
        stride1[i] = 0;
        stride2[i] = 0;
    }
    for (int i = 1; i < length + 1; ++i) {
        if (size1.data[length - i] == outSize.data[len - i]) {
            stride1[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
    length = size2.len;
    stride = 1;
    for (int i = 1; i < length + 1; ++i) {
        if (size2.data[length - i] == outSize.data[len - i]) {
            stride2[len - i] = stride;
            stride *= outSize.data[len - i];
        }
    }
}

extern "C" diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOther = impl::cuda::makeTensor(other);
    auto trOut   = impl::cuda::makeTensor(out);

    int blockSize = 256;
    double coff = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
    } else {
        coff = alpha->fval;
    }
    diopiSize_t inShape = trInput.shape();
    diopiSize_t othShape = trOther.shape();
    int gridSize  = (trOut.numel() + blockSize - 1) / blockSize;
    if (compareShape(inShape, othShape)) {
        DISPATCH_DTYPE(vecAdd, trInput.dtype(), gridSize, blockSize, stream,
            trInput.data(), trOther.data(), trOut.data(), trInput.numel(), coff);
    } else {
        diopiSize_t outShape = trOut.shape();
        diopiSize_t outStrideHost = trOut.stride();
        int len = outShape.len;
        int64_t nbytes = len * sizeof(int64_t);

        std::vector<int64_t> inStrideHost(len);
        std::vector<int64_t> othStrideHost(len);
        auto inStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto othStride = impl::cuda::requiresBuffer(ctx, nbytes);
        auto outStride = impl::cuda::requiresBuffer(ctx, nbytes);

        computeStride(inShape, othShape, outShape, inStrideHost.data(), othStrideHost.data());
        cudaMemcpyAsync(inStride.data(), inStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(othStride.data(), othStrideHost.data(), nbytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(outStride.data(), outStrideHost.data, nbytes, cudaMemcpyHostToDevice, stream);

        DISPATCH_DTYPE(vecAddBroadcast, trInput.dtype(), gridSize, blockSize, stream,
           trInput.data(), trOther.data(), trOut.data(), trOut.numel(), coff, static_cast<const int64_t*>(inStride.data()),
           static_cast<const int64_t*>(othStride.data()), static_cast<const int64_t*>(outStride.data()), len);
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    auto stream  = impl::cuda::getStream(ctx);
    auto trInput = impl::cuda::makeTensor(input);
    auto trOut   = impl::cuda::makeTensor(out);
    int blockSize = 256;
    double coff = 0.0;
    double otherVal = 0.0;
    if (trInput.dtype() <= 7) {
        coff = alpha->ival;
        otherVal = other->ival;
    } else {
        coff = alpha->fval;
        otherVal = other->fval;
    }
    int gridSize = (trInput.numel() + blockSize - 1) / blockSize;
    DISPATCH_DTYPE(vecAddScalar, trInput.dtype(), gridSize, blockSize, stream,
        trInput.data(), otherVal, trOut.data(), trInput.numel(), coff);
    return diopiSuccess;
}

template<typename T> __global__
void vecFill(void* a, const float value, const int numel) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    T* A = static_cast<T*>(a);
    if (id < numel) {
        A[id] = static_cast<T>(value);
    }
}

extern "C" diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream = impl::cuda::getStream(ctx);
    auto tr = impl::cuda::makeTensor(input);

    diopiDevice_t device = tr.device();
    diopiDtype_t  dtype  = tr.dtype();
    int64_t       numel  = tr.numel();
    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }
    if (diopi_host == device) {
        return diopiErrorOccurred;
    } else {
        int blockSize = 256;
        int gridSize  = (numel + blockSize - 1) / blockSize;
        DISPATCH_DTYPE(vecFill, dtype, gridSize, blockSize, stream, tr.data(), val, numel);
    }

    return diopiSuccess;
}
