/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <acl/acl.h>
#include <diopi/diopirt.h>
#include <diopi_register.h>

#include <cstdio>

#include "error.hpp"

namespace impl {
namespace ascend {

#define CALL_ACLRT(Expr)                                                                                          \
    {                                                                                                             \
        ::aclError ret = Expr;                                                                                    \
        if (ret != ::ACL_SUCCESS) {                                                                               \
            printf("call a ascendrt function (%s) failed. return code=%d, %s", #Expr, ret, aclGetRecentErrMsg()); \
        }                                                                                                         \
    }

extern "C" {
void* ascend_malloc(uint64_t bytes) {
    void* ptr;
    CALL_ACLRT(::aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

void ascend_free(void* ptr) { CALL_ACLRT(aclrtFree(ptr)); }

int32_t ascend_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    CALL_ACLRT(aclrtCreateStream((aclrtStream*)stream_handle_ptr));
    return diopiSuccess;
}

int32_t ascend_destroy_stream(diopiStreamHandle_t stream_handle) {
    CALL_ACLRT(aclrtDestroyStream((aclrtStream)stream_handle));
    return diopiSuccess;
}

int32_t ascend_synchronize_stream(diopiStreamHandle_t stream_handle) {
    CALL_ACLRT(aclrtSynchronizeStream((aclrtStream)stream_handle));
    return diopiSuccess;
}

int32_t ascend_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)stream_handle));
    return diopiSuccess;
}

int32_t ascend_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, (aclrtStream)stream_handle));
    return diopiSuccess;
}

int32_t ascend_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, (aclrtStream)stream_handle));
    return diopiSuccess;
}

int32_t initLibrary() {
    CALL_ACLRT(aclInit(nullptr));
    CALL_ACLRT(aclrtSetDevice(0));
    diopiRegisterDeviceMallocFunc(ascend_malloc);
    diopiRegisterDevMemFreeFunc(ascend_free);
    diopiRegisterStreamCreateFunc(ascend_make_stream);
    diopiRegisterStreamDestroyFunc(ascend_destroy_stream);
    diopiRegisterSynchronizeStreamFunc(ascend_synchronize_stream);
    diopiRegisterMemcpyD2HAsyncFunc(ascend_memcpy_d2h_async);
    diopiRegisterMemcpyD2DAsyncFunc(ascend_memcpy_d2d_async);
    diopiRegisterMemcpyH2DAsyncFunc(ascend_memcpy_h2d_async);
    diopiRegisterGetLastErrorFunc(ascend_get_last_error_string);

    return diopiSuccess;
}

int32_t finalizeLibrary() {
    CALL_ACLRT(aclFinalize());
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
