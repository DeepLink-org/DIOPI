/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <conform_test.h>
#include <diopi/diopirt.h>

#include <cstdio>

#include "../ascend_tensor.hpp"
#include "../error.hpp"

namespace impl {
namespace ascend {

extern "C" {
void* device_malloc(uint64_t bytes) {
    void* ptr = nullptr;
    if (bytes > 0) {
        CALL_ACLRT(::aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    return ptr;
}

void device_free(void* ptr) {
    if (ptr) {
        CALL_ACLRT(aclrtFree(ptr));
    }
}

diopiError_t device_make_stream(diopiStreamHandle_t* streamHandlePtr) {
    CALL_ACLRT(aclrtCreateStream(reinterpret_cast<aclrtStream*>(streamHandlePtr)));
    return diopiSuccess;
}

diopiError_t device_destroy_stream(diopiStreamHandle_t streamHandle) {
    CALL_ACLRT(aclrtDestroyStream(reinterpret_cast<aclrtStream>(streamHandle)));
    return diopiSuccess;
}

diopiError_t device_synchronize_stream(diopiStreamHandle_t streamHandle) {
    CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(streamHandle)));
    return diopiSuccess;
}

diopiError_t device_memcpy_h2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    if (nullptr != dst && nullptr != src) {
        CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, reinterpret_cast<aclrtStream>(streamHandle)));
    }
    return diopiSuccess;
}

diopiError_t device_memcpy_d2h_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    if (nullptr != dst && nullptr != src) {
        CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, reinterpret_cast<aclrtStream>(streamHandle)));
    }
    return diopiSuccess;
}

diopiError_t device_memcpy_d2d_async(diopiStreamHandle_t streamHandle, void* dst, const void* src, uint64_t bytes) {
    CALL_ACLRT(aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, reinterpret_cast<aclrtStream>(streamHandle)));
    return diopiSuccess;
}

diopiError_t initLibrary() {
    CALL_ACLRT(aclInit(nullptr));
    CALL_ACLRT(aclrtSetDevice(0));
    aclrtContext context;
    CALL_ACLRT(aclrtCreateContext(&context, 0));
    return diopiSuccess;
}

diopiError_t finalizeLibrary() {
    CALL_ACLRT(aclFinalize());
    return diopiSuccess;
}

// temporary solution, which needs to be re-implemented later
diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) { return diopiSuccess; }

}  // extern "C"

}  // namespace ascend
}  // namespace impl
