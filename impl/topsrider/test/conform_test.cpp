/**************************************************************************************************
 * Copyright 2022 Enflame. All Rights Reserved.
 * License: BSD 3-Clause
 * Author: boris.wu
 *
 *************************************************************************************************/
#include <diopi/diopirt.h>

#include <tops/tops_ext.h>

#include <cstdio>
#include <mutex>

#include "log.h"
#include "ops.h"

#define CALL_TOPS(Expr)                                                               \
    {                                                                                 \
        topsError_t ret = Expr;                                                       \
        if (ret != topsSuccess) {                                                     \
            printf("call a topsrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }

extern "C" {

void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_TOPS(topsMallocScatter(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) { CALL_TOPS(topsFree(ptr)); }

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    topsStream_t ph_stream;
    CALL_TOPS(topsStreamCreate(&ph_stream));
    *stream_handle_ptr = (diopiStreamHandle_t)ph_stream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    auto* ph_stream = (topsStream_t)stream_handle;
    CALL_TOPS(topsStreamDestroy(ph_stream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    auto* ph_stream = (topsStream_t)stream_handle;
    CALL_TOPS(topsStreamSynchronize(ph_stream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    auto* ph_stream = (topsStream_t)stream_handle;
    CALL_TOPS(topsMemcpyAsync(dst, src, bytes, topsMemcpyHostToDevice, ph_stream));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    auto* ph_stream = (topsStream_t)stream_handle;
    CALL_TOPS(topsMemcpyAsync(dst, src, bytes, topsMemcpyDeviceToHost, ph_stream));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    auto* ph_stream = (topsStream_t)stream_handle;
    CALL_TOPS(topsMemcpyAsync(dst, src, bytes, topsMemcpyDeviceToDevice, ph_stream));
    return diopiSuccess;
}

int32_t initLibrary() {
    // impl::tops::topsLibInit();

    /*
      diopiRegisterDeviceMallocFunc(device_malloc);
      diopiRegisterDevMemFreeFunc(device_free);
      diopiRegisterStreamCreateFunc(device_make_stream);
      diopiRegisterStreamDestroyFunc(device_destroy_stream);
      diopiRegisterSynchronizeStreamFunc(device_synchronize_stream);
      diopiRegisterMemcpyD2HAsyncFunc(device_memcpy_d2h_async);
      diopiRegisterMemcpyD2DAsyncFunc(device_memcpy_d2d_async);
      diopiRegisterMemcpyH2DAsyncFunc(device_memcpy_h2d_async);
      diopiRegisterGetLastErrorFunc(device_get_last_error_string);
     */

    return diopiSuccess;
}

int32_t finalizeLibrary() {
    // impl::tops::topsLibFinalize();
    return diopiSuccess;
}

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) { return diopiSuccess; }

}  // extern "C"
