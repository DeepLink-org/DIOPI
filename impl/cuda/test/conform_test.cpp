/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cuda_runtime.h>
#include <diopi/diopirt.h>

#include <cstdio>
#include <mutex>

extern "C" {

#define CALL_CUDA(Expr)                                                               \
    {                                                                                 \
        cudaError_t ret = Expr;                                                       \
        if (ret != cudaSuccess) {                                                     \
            printf("call a cudart function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                             \
    }

void* device_malloc(uint64_t bytes) {
    void* ptr;
    CALL_CUDA(cudaMalloc(&ptr, bytes));
    return ptr;
}

void device_free(void* ptr) { CALL_CUDA(cudaFree(ptr)); }

int32_t device_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
    cudaStream_t phStream;
    CALL_CUDA(cudaStreamCreate(&phStream));
    *stream_handle_ptr = (diopiStreamHandle_t)phStream;
    return diopiSuccess;
}

int32_t device_destroy_stream(diopiStreamHandle_t stream_handle) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaStreamDestroy(phStream));
    return diopiSuccess;
}

int32_t device_synchronize_stream(diopiStreamHandle_t stream_handle) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaStreamSynchronize(phStream));
    return diopiSuccess;
}

int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, phStream));
    return diopiSuccess;
}

int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, phStream));
    return diopiSuccess;
}

int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst, const void* src, uint64_t bytes) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, phStream));
    return diopiSuccess;
}

static char strLastError[8192] = {0};
static char strLastErrorOther[4096] = {0};
static std::mutex mtxLastError;

const char* device_get_last_error_string() {
    cudaError_t error = cudaGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "cuda error: %s; other error: %s", cudaGetErrorString(error), strLastErrorOther);
    return strLastError;
}

diopiError_t initLibrary() { return diopiSuccess; }

diopiError_t finalizeLibrary() { return diopiSuccess; }

}  // extern "C"

namespace impl {

namespace cuda {

void _set_last_error_string(const char* err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}

}  // namespace cuda

}  // namespace impl
