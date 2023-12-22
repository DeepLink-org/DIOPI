#include <conform_test.h>
#include <diopi/diopirt.h>

#include <cstdio>

#include "../common/common.hpp"
#include "../error.hpp"
#include "litert.hpp"

namespace impl {
namespace kunlunxin {

extern "C" {

void* device_malloc(uint64_t bytes) {
    void* ptr;
    xpu_malloc(&ptr, bytes);
    return ptr;
}

void device_free(void* ptr) { xpu_free(ptr); }

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    XPUStream pstream = nullptr;
    xpu_stream_create(&pstream);
    *stream_handle_ptr = (diopiStreamHandle_t)pstream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    XPUStream pstream = (XPUStream)stream_handle;
    xpu_stream_destroy(pstream);
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    XPUStream pstream = (XPUStream)stream_handle;
    xpu_wait(pstream);
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    XPUStream pstream = (XPUStream)stream_handle;
    xpu_wait(pstream);
    xpu_memcpy(dst, src, bytes, XPU_HOST_TO_DEVICE);
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    XPUStream pstream = (XPUStream)stream_handle;
    xpu_wait(pstream);
    xpu_memcpy(dst, src, bytes, XPU_DEVICE_TO_HOST);
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) { return diopiNotInited; }

int32_t initLibrary() { return diopiSuccess; }

int32_t finalizeLibrary() { return diopiSuccess; }

diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out) {
    std::vector<int64_t> vec{808};
    diopiSize_t size{vec.data(), static_cast<int64_t>(vec.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, diopi_dtype_uint8, diopi_host);
    *out = *tensor;
    return diopiSuccess;
}

}  // extern "C"

}  // namespace kunlunxin
}  // namespace impl
