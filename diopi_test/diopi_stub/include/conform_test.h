/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 * @brief A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 */

#ifndef IMPL_CONFORM_TEST_H_  // NOLINT
#define IMPL_CONFORM_TEST_H_  // NOLINT

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * typedef for functions
 **/

// typedef int32_t (*create_stream_func_t)(diopiStreamHandle_t*);
// typedef int32_t (*destroy_stream_func_t)(diopiStreamHandle_t);

typedef void* (*malloc_func_t)(uint64_t);
typedef void (*free_func_t)(void*);

// typedef int32_t (*memcpy_h2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
// typedef int32_t (*memcpy_d2h_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
// typedef int32_t (*memcpy_d2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);

// typedef int32_t (*sync_stream_func_t)(diopiStreamHandle_t stream);

// typedef const char* (*get_last_error_string_func_t)();

/**
 * operations to register user-provided functions
 **/
// extern DIOPI_API diopiError_t diopiRegisterStreamCreateFunc();
// extern DIOPI_API diopiError_t diopiRegisterStreamDestroyFunc();
// extern DIOPI_API diopiError_t diopiRegisterSynchronizeStreamFunc();

// extern DIOPI_API diopiError_t diopiRegisterDeviceMallocFunc();
// extern DIOPI_API diopiError_t diopiRegisterDevMemFreeFunc();

// extern DIOPI_API diopiError_t diopiRegisterMemcpyH2DAsyncFunc();
// extern DIOPI_API diopiError_t diopiRegisterMemcpyD2HAsyncFunc();
// extern DIOPI_API diopiError_t diopiRegisterMemcpyD2DAsyncFunc();

// extern DIOPI_API diopiError_t diopiRegisterGetLastErrorFunc();

extern void* device_malloc(uint64_t bytes);
extern void device_free(void* ptr);
extern int32_t device_memcpy_h2d_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern int32_t device_memcpy_d2h_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern int32_t device_memcpy_d2d_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern int32_t device_make_stream(diopiStreamHandle_t*);
extern int32_t device_destroy_stream(diopiStreamHandle_t);
extern int32_t device_synchronize_stream(diopiStreamHandle_t stream);
/**
 * User-implemented functions
 **/
extern int32_t initLibrary();
extern int32_t finalizeLibrary();

extern int32_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out);

#if defined(__cplusplus)
}
#endif

#endif  // IMPL_CONFORM_TEST_H  // NOLINT
