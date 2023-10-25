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

#include <cstdio>
#include <mutex>

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

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    tangStream_t phStream;
    CALL_DROPLET(tangStreamCreate(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangStreamDestroy(phStream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangStreamSynchronize(phStream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangMemcpyAsync(dst, src, bytes, tangMemcpyHostToDevice, phStream));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    tangStream_t phStream = (tangStream_t)stream_handle;
    CALL_DROPLET(tangMemcpyAsync(dst, src, bytes, tangMemcpyDeviceToHost, phStream));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
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

int32_t finalizeLibrary() { return diopiSuccess; }

int32_t initLibrary() { return diopiSuccess; }

#if defined(__cplusplus)
}  // extern "C"
#endif
