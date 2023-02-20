/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <diopi/diopirt.h>
#include <diopi_register.h>
#include <cnnl.h>
#include <cnmlrt.h>

#include <cstdio>
#include <mutex>

#include "helper.hpp"


extern "C" {


#define CALL_CAMB(Expr)   {                                                         \
    ::cnrtRet_t ret = Expr;                                                         \
    if (ret != ::CNRT_RET_SUCCESS) {                                                \
        printf("call a cambrt function (%s) failed. return code=%d", #Expr, ret);   \
    }}                                                                              \

void* camb_malloc(uint64_t bytes) {
    void* ptr;
    CALL_CAMB(::cnrtMalloc(&ptr, bytes));
    return ptr;
}

void camb_free(void* ptr) {
    CALL_CAMB(cnrtFree(ptr));
}

int32_t camb_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    cnrtQueue_t phStream;
    CALL_CAMB(cnrtCreateQueue(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t camb_destroy_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(cnrtDestroyQueue(phStream));
    return diopiSuccess;
}

int32_t camb_synchronize_stream(diopiStreamHandle_t stream_handle) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(cnrtSyncQueue(phStream));
    return diopiSuccess;
}

int32_t camb_memcpy_h2d_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV));
    return diopiSuccess;
}

int32_t camb_memcpy_d2h_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2HOST));
    return diopiSuccess;
}

int32_t camb_memcpy_d2d_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes) {
    cnrtQueue_t phStream = (cnrtQueue_t)stream_handle;
    CALL_CAMB(cnrtMemcpyAsync(dst, const_cast<void *>(src), bytes, phStream, CNRT_MEM_TRANS_DIR_DEV2DEV));
    return diopiSuccess;
}

static char strLastError[8192] = {0};
static char strLastErrorOther[4096] = {0};
static std::mutex mtxLastError;

const char* camb_get_last_error_string() {
    // consider cnrt version cnrtGetLastErr or cnrtGetLastError
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "camb error: %s; other error: %s",
            ::cnrtGetErrorStr(err), strLastErrorOther);
    return strLastError;
}

int32_t initLibrary() {
    diopiRegisterDeviceMallocFunc(camb_malloc);
    diopiRegisterDevMemFreeFunc(camb_free);
    diopiRegisterStreamCreateFunc(camb_make_stream);
    diopiRegisterStreamDestroyFunc(camb_destroy_stream);
    diopiRegisterSynchronizeStreamFunc(camb_synchronize_stream);
    diopiRegisterMemcpyD2HAsyncFunc(camb_memcpy_d2h_async);
    diopiRegisterMemcpyD2DAsyncFunc(camb_memcpy_d2d_async);
    diopiRegisterMemcpyH2DAsyncFunc(camb_memcpy_h2d_async);
    diopiRegisterGetLastErrorFunc(camb_get_last_error_string);

    return diopiSuccess;
}

int32_t finalizeLibrary() {
    return diopiSuccess;
}

}  // extern "C"

namespace impl {

namespace camb {

void _set_last_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}

}  // namespace camb

}  // namespace impl
