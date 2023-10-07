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

typedef void* (*malloc_func_t)(uint64_t);
typedef void (*free_func_t)(void*);

/**
 * operations to register user-provided functions
 **/

extern void* device_malloc(uint64_t bytes);
extern void device_free(void* ptr);
extern diopiError_t device_memcpy_h2d_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern diopiError_t device_memcpy_d2h_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern diopiError_t device_memcpy_d2d_async(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
extern diopiError_t device_make_stream(diopiStreamHandle_t*);
extern diopiError_t device_destroy_stream(diopiStreamHandle_t);
extern diopiError_t device_synchronize_stream(diopiStreamHandle_t stream);

/**
 * User-implemented functions
 **/
extern diopiError_t initLibrary();
extern diopiError_t finalizeLibrary();

extern diopiError_t buildGeneratorState(diopiContextHandle_t ctx, diopiTensorHandle_t out);

#if defined(__cplusplus)
}
#endif

#endif  // IMPL_CONFORM_TEST_H  // NOLINT
