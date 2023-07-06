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
void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_ACLRT(::aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
}

void device_free(void* ptr) { CALL_ACLRT(aclrtFree(ptr)); }

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    CALL_ACLRT(aclrtCreateStream(reinterpret_cast<aclrtStream*>(stream_handle_ptr)));
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    CALL_ACLRT(aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream_handle)));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_handle)));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(stream_handle)));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, reinterpret_cast<aclrtStream>(stream_handle)));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, reinterpret_cast<aclrtStream>(stream_handle)));
    return diopiSuccess;
}

int32_t finalizeLibrary() {
    CALL_ACLRT(aclFinalize());
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
