/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnmlrt.h>
#include <cnnl.h>
#include <conform_test.h>
#include <diopi/diopirt.h>
#include "litert.hpp"

#include <cstdio>
#include <mutex>
#include <vector>
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

int32_t device_make_stream(diopiStreamHandle_t* streamHandlePtr) {
    cnrtQueue_t phStream;
    CALL_CNRT(cnrtCreateQueue(&phStream));
    *streamHandlePtr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t streamHandle) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtDestroyQueue(phStream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t streamHandle) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtSyncQueue(phStream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    CALL_CNRT(cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
    return diopiSuccess;
}

int32_t initLibrary() { return diopiSuccess; }

int32_t finalizeLibrary() { return diopiSuccess; }

int32_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    std::vector<int64_t> vec{1180672};
    diopiSize_t size{vec.data(), static_cast<int64_t>(vec.size())};
    diopiTensorHandle_t new_tensor = nullptr;
    diopiRequireTensor(ctx, &new_tensor, &size, nullptr, diopi_dtype_uint8, diopi_device);
    *out = *new_tensor;
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
