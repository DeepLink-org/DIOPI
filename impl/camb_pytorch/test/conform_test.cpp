/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cnnl.h>
#include <cnrt.h>
#include <diopi/diopirt.h>

#include <cstdio>
#include <mutex>

#include "error.hpp"

extern "C" {

#define CALL_CAMB(Expr)                                                               \
    {                                                                                 \
        ::cnrtRet_t ret = Expr;                                                       \
        if (ret != ::CNRT_RET_SUCCESS) {                                              \
            printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }

void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_CAMB(::cnrtMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) { CALL_CAMB(::cnrtFree(ptr)); }

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    cnrtQueue_t phStream;
    CALL_CAMB(::cnrtQueueCreate(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(::cnrtQueueDestroy(phStream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(::cnrtQueueSync(phStream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(::cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(::cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(::cnrtMemcpyAsync(dst, const_cast<void*>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
    return diopiSuccess;
}

int32_t initLibrary() { return diopiSuccess; }

int32_t finalizeLibrary() { return diopiSuccess; }

}  // extern "C"