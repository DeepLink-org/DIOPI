/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#include <acl/acl_base.h>
#include <conform_test.h>
#include <diopi/diopirt.h>

#include <cstdio>
#include <map>
#include <string>

#include "ascend_helper.hpp"
#include "litert.hpp"

#include "acl/acl.h"
// #include "all_ops.h"
#include "graph/ascend_string.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_ir_build.h"
#include "graph/gnode.h"
#include "graph/graph.h"
// #include "litert.hpp"
// #include "tensor.h"
#include "graph/types.h"
using namespace ge;


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
    ASCEND_CHECK_NULLPTR_ABORT(*streamHandlePtr);
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
    CALL_ACLRT(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
    // aclrtContext context;
    // CALL_ACLRT(aclrtCreateContext(&context, 0));
    
    std::string path = "/home/code/DIOPI/diopi_test/python/fusion_switch_file.cfg";
    // std::map<std::string, std::string> global_options = {
    //     {std::string(ge::ir_option::FUSION_SWITCH_FILE), std::string(path.c_str())},
    // };
    // CALL_ACLRT(ge::aclgrphBuildInitialize(global_options));
    std::cout << "ascend_npu success initLibrary" << std::endl;
    return diopiSuccess;
}

diopiError_t finalizeLibrary() {
    CALL_ACLRT(aclFinalize());
    return diopiSuccess;
}

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    // state size = seed size + offset size
    std::vector<int64_t> vec{sizeof(uint64_t) + sizeof(int64_t)};
    diopiSize_t size{vec.data(), static_cast<int64_t>(vec.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, diopi_dtype_uint8, diopi_host);
    *out = *tensor;
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
