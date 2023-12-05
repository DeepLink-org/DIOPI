#include <diopi/diopirt.h>

#include <cstdio>

#include "../common/common.hpp"
#include "../error.hpp"

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

}  // extern "C"

}  // namespace kunlunxin
}  // namespace impl
