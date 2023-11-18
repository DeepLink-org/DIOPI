/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <conform_test.h>
#include <diopi/diopirt.h>
#include <supa.h>

#include <cassert>
#include <iostream>
#include <mutex>

#define SUPA_CALL(Expr)                                                                                                                       \
    {                                                                                                                                         \
        suError_t ret = Expr;                                                                                                                 \
        if (ret != suSuccess) {                                                                                                               \
            std::cout << "Supa function (" << #Expr << ") failed. return code=" << ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        }                                                                                                                                     \
    }

extern "C" {
DIOPI_API void* br_device_malloc(uint64_t bytes);
DIOPI_API void br_device_free(void* ptr);
DIOPI_API void* get_phy_ptr(const void* ptr);
}

void* device_malloc(uint64_t bytes) {
    if (br_device_malloc) {
        return br_device_malloc(bytes);
    }
    void* ptr = nullptr;
    SUPA_CALL(suMallocDevice(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) {
    if (br_device_free) {
        br_device_free(ptr);
        return;
    }
    SUPA_CALL(suFree(ptr));
}

diopiError_t device_make_stream(diopiStreamHandle_t* streamHandlePtr) {
    suStream_t stream = nullptr;
    SUPA_CALL(suStreamCreate(&stream));
    *streamHandlePtr = (diopiStreamHandle_t)stream;
    return diopiSuccess;
}

diopiError_t device_destroy_stream(diopiStreamHandle_t streamHandle) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suStreamDestroy(stream));
    return diopiSuccess;
}

diopiError_t device_synchronize_stream(diopiStreamHandle_t streamHandle) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suStreamSynchronize(stream));
    return diopiSuccess;
}

diopiError_t device_memcpy_h2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    auto phy_gpu_addr = get_phy_ptr(dst); // the gpu address may be a virtual address. so translate it.
    SUPA_CALL(suMemcpyAsync(phy_gpu_addr, const_cast<void*>(src), bytes, stream, suMemcpyHostToDevice));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2h_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    auto phy_gpu_addr = get_phy_ptr(src);
    SUPA_CALL(suMemcpyAsync(dst, const_cast<void*>(phy_gpu_addr), bytes, stream, suMemcpyDeviceToHost));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    auto phy_src_gpu_addr = get_phy_ptr(src);
    auto phy_dst_gpu_addr = get_phy_ptr(dst);
    SUPA_CALL(suMemcpyAsync(phy_dst_gpu_addr, const_cast<void*>(phy_src_gpu_addr), bytes, stream, suMemcpyDeviceToDevice));
    return diopiSuccess;
}

diopiError_t initLibrary() { return diopiSuccess; }

diopiError_t finalizeLibrary() { return diopiSuccess; }

#include "litert.hpp"

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    const int64_t size_data[] = {2, 2};
    diopiSize_t size{size_data, sizeof(size_data) / sizeof(size_data[0])};
    diopiTensorHandle_t state = nullptr;
    diopiRequireTensor(ctx, &state, &size, nullptr, diopi_dtype_int32, diopi_device);
    *out = *state;
    return diopiSuccess;
}

diopiError_t diopiTensorCopyToBuffer(diopiContextHandle_t ctx, diopiConstTensorHandle_t tensor, void* dst) {
    if (tensor->device() == diopi_device) {
        diopiTensorHandle_t dst_tensor;
        diopiSize_t stride;
        diopiDtype_t dtype;
        diopiSize_t size;
        diopiGetTensorDtype(tensor, &dtype);
        diopiGetTensorShape(tensor, &size);
        diopiGetTensorStride(tensor, &stride);
        diopiRequireTensor(ctx, &dst_tensor, &size, &stride, dtype, diopiDevice_t::diopi_host);
        // D2H
        diopiCopyInp(ctx, tensor, dst_tensor);
        std::memcpy(dst, dst_tensor->data(), tensor->nbytes());
    } else {
        std::memcpy(dst, tensor->data(), tensor->nbytes());
    }
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiTensorCopyFromBuffer(diopiContextHandle_t ctx, const void* src, diopiTensorHandle_t tensor) {
    if (tensor->device() == diopi_device) {
        diopiStreamHandle_t stream;
        diopiTensorHandle_t dst_tensor;
        diopiSize_t stride;
        diopiDtype_t dtype;
        diopiSize_t size;
        diopiGetTensorDtype(tensor, &dtype);
        diopiGetTensorShape(tensor, &size);
        diopiGetTensorStride(tensor, &stride);
        diopiRequireTensor(ctx, &dst_tensor, &size, &stride, dtype, diopiDevice_t::diopi_host);

        diopiGetStream(ctx, &stream);
        std::memcpy(dst_tensor->data(), src, tensor->nbytes());
        device_synchronize_stream(stream);
        // H2D
        diopiCopyInp(ctx, dst_tensor, tensor);
        device_synchronize_stream(stream);
    } else {
        std::memcpy(tensor->data(), src, tensor->nbytes());
    }
    return diopiSuccess;
}
