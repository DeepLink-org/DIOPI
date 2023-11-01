/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <conform_test.h>
#include <diopi/diopirt.h>
#include <tang_compiler_api.h>
#include <tang_runtime.h>
#include <vector>

#include <cstdio>
#include <mutex>

#include "litert.hpp"

#if defined(__cplusplus)
extern "C" {
#endif

#define CALL_DROPLET(Expr)                                                            \
    {                                                                                 \
        tangError_t ret = Expr;                                                       \
        if (ret != tangSuccess) {                                                     \
            printf("call a tangrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }

void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_DROPLET(tangMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) { CALL_DROPLET(tangFree(ptr)); }

diopiError_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    tangStream_t phStream;
    CALL_DROPLET(tangStreamCreate(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

diopiError_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangStreamDestroy(phStream));
    return diopiSuccess;
}

diopiError_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangStreamSynchronize(phStream));
    return diopiSuccess;
}

diopiError_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangMemcpyAsync(dst, src, bytes, tangMemcpyHostToDevice, phStream));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangMemcpyAsync(dst, src, bytes, tangMemcpyDeviceToHost, phStream));
    return diopiSuccess;
}

diopiError_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangMemcpyAsync(dst, src, bytes, tangMemcpyDeviceToDevice, phStream));
    return diopiSuccess;
}

static char strLastError[8192] = {0};
static char strLastErrorOther[4096] = {0};
static std::mutex mtxLastError;

const char* device_get_last_error_string() {
    tangError_t error = tangGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "droplet error: %s; other error: %s", tangGetErrorString(error), strLastErrorOther);
    return strLastError;
}

diopiError_t finalizeLibrary() { return diopiSuccess; }

diopiError_t initLibrary() { return diopiSuccess; }

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    std::vector<int64_t> vec{808};
    diopiSize_t size{vec.data(), static_cast<int64_t>(vec.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, diopi_dtype_uint8, diopi_host);
    *out = *tensor;
    return diopiSuccess;
}

#if defined(__cplusplus)
}  // extern "C"
#endif
