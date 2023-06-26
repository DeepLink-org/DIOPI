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

#define SUPA_CALL(Expr)                                                                              \
    {                                                                                                \
        suError_t ret = Expr;                                                                        \
        if (ret != suSuccess) {                                                                      \
            std::cout << "Supa function (" << #Expr << ") failed. return code=" << ret << std::endl; \
        }                                                                                            \
    }

extern "C" {
DIOPI_API void* br_device_malloc(uint64_t bytes);
DIOPI_API void br_device_free(void* ptr);
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

int32_t device_make_stream(diopiStreamHandle_t* streamHandlePtr) {
    suStream_t stream = nullptr;
    SUPA_CALL(suStreamCreate(&stream));
    *streamHandlePtr = (diopiStreamHandle_t)stream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t streamHandle) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suStreamDestroy(stream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t streamHandle) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suStreamSynchronize(stream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suMemcpyAsync(dst, const_cast<void*>(src), bytes, stream, suMemcpyHostToDevice));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suMemcpyAsync(dst, const_cast<void*>(src), bytes, stream, suMemcpyDeviceToHost));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    suStream_t stream = (suStream_t)streamHandle;
    SUPA_CALL(suMemcpyAsync(dst, const_cast<void*>(src), bytes, stream, suMemcpyDeviceToDevice));
    return diopiSuccess;
}

int32_t finalizeLibrary() { return diopiSuccess; }
