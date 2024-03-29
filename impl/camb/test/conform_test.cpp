/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnmlrt.h>
#include <cnnl.h>
#include <conform_test.h>
#include <diopi/diopirt.h>

#include <cstdio>
#include <mutex>
#include <vector>

#include "litert.hpp"
namespace impl {
namespace camb {

#define CALL_CNRT(Expr)                                                               \
    {                                                                                 \
        ::cnrtRet_t ret = Expr;                                                       \
        if (ret != ::CNRT_RET_SUCCESS) {                                              \
            printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }

extern "C" {
void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_CNRT(::cnrtMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) { CALL_CNRT(cnrtFree(ptr)); }

diopiError_t device_make_stream(diopiStreamHandle_t* streamHandlePtr) {
    cnrtQueue_t phStream;
    CALL_CNRT(cnrtCreateQueue(&phStream));
    *streamHandlePtr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

diopiError_t device_destroy_stream(diopiStreamHandle_t streamHandle) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtDestroyQueue(phStream));
    return diopiSuccess;
}

diopiError_t device_synchronize_stream(diopiStreamHandle_t streamHandle) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtSyncQueue(phStream));
    return diopiSuccess;
}

diopiError_t device_memcpy_h2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2h_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
    return diopiSuccess;
}

diopiError_t initLibrary() { return diopiSuccess; }

diopiError_t finalizeLibrary() { return diopiSuccess; }

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    size_t stateSize = 0;
    cnnlRandGetMTGP32StateSize(nullptr, &stateSize);
    std::vector<int64_t> vec{static_cast<int64_t>(stateSize)};
    diopiSize_t size{vec.data(), static_cast<int64_t>(vec.size())};
    diopiTensorHandle_t state = nullptr;
    diopiRequireTensor(ctx, &state, &size, nullptr, diopi_dtype_uint8, diopi_device);

    void* statePtr = nullptr;
    diopiGetTensorData(state, &statePtr);
    uint32_t seed = 0;
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    diopiStreamHandle_t streamHandle;
    diopiGetStream(ctx, &streamHandle);
    cnrtQueue_t queue = static_cast<cnrtQueue_t>(streamHandle);
    cnnlSetQueue(handle, queue);
    cnnlRandMakeMTGP32KernelState(handle, statePtr, nullptr, nullptr, seed);

    *out = *state;
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
