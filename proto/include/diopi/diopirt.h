/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_RT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_RT_H_

#include <stdint.h>

#ifndef DIOPI_ATTR_WEAK
#define DIOPI_API
#define DIOPI_RT_API
#else
#define DIOPI_API __attribute__((weak))
#define DIOPI_RT_API __attribute__((weak))
#endif

#if defined(__cplusplus)
#include <iostream>
extern "C" {
#endif

#define DIOPI_VER_MAJOR 1
#define DIOPI_VER_MINOR 0
#define DIOPI_VER_PATCH 0
#define DIOPI_VERSION (DIOPI_VER_MAJOR * 1000 + DIOPI_VER_MINOR * 100 + DIOPI_VER_PATCH)

typedef struct {
    const int64_t* data;
    int64_t len;
} diopiSize_t;

typedef enum {
    diopiSuccess = 0,
    diopiErrorOccurred = 1,
    diopiNotInited = 2,
    diopiNoRegisteredStreamCreateFunction = 3,
    diopiNoRegisteredStreamDestoryFunction = 4,
    diopiNoRegisteredStreamSyncFunction = 5,
    diopiNoRegisteredDeviceMemoryMallocFunction = 6,
    diopiNoRegisteredDeviceMemoryFreeFunction = 7,
    diopiNoRegisteredDevice2DdeviceMemoryCopyFunction = 8,
    diopiNoRegisteredDevice2HostMemoryCopyFunction = 9,
    diopiNoRegisteredHost2DeviceMemoryCopyFunction = 10,
    diopiNoRegisteredGetLastErrorFunction = 11,
    diopi5DNotSupported = 12,
    diopiNoImplement = 13,
    diopiDtypeNotSupported = 1000,
} diopiError_t;

typedef enum {
    diopi_host = 0,
    diopi_device = 1,
} diopiDevice_t;

typedef enum {
    diopi_dtype_int8 = 0,
    diopi_dtype_uint8 = 1,
    diopi_dtype_int16 = 2,
    diopi_dtype_uint16 = 3,
    diopi_dtype_int32 = 4,
    diopi_dtype_uint32 = 5,
    diopi_dtype_int64 = 6,
    diopi_dtype_uint64 = 7,
    diopi_dtype_float16 = 8,
    diopi_dtype_float32 = 9,
    diopi_dtype_float64 = 10,
    diopi_dtype_bool = 11,
    diopi_dtype_bfloat16 = 12,
    diopi_dtype_tfloat32 = 13,
    diopi_dtype_complex32 = 14,
    diopi_dtype_complex64 = 15,
    diopi_dtype_complex128 = 16,
    diopi_dtype_unsupported = 255
} diopiDtype_t;

typedef struct {
    diopiDtype_t stype;
    union {
        double fval;
        int64_t ival;
    };
} diopiScalar_t;

typedef enum { Contiguous = 0, ChannelsLast = 1, ChannelsLast3d = 2, Preserve = 3, ChannelsLast1d = 4 } diopiMemoryFormat_t;

typedef enum { ReductionNone, ReductionMean, ReductionSum, ReductionEND } diopiReduction_t;

typedef enum { RoundModeNone, RoundModeTrunc, RoundModeFloor, RoundModeEND } diopiRoundMode_t;

/**
 * Opaque structure holding Context and Tensor
 **/
struct diopiContext;
typedef struct diopiContext* diopiContextHandle_t;

struct diopiTensor;
typedef struct diopiTensor* diopiTensorHandle_t;
typedef const struct diopiTensor* diopiConstTensorHandle_t;

/**
 * Opaque pointer of Stream
 **/
typedef void* diopiStreamHandle_t;

/**
 * get the version of the Device-Independent Operator Inetrface
 */
extern DIOPI_API const char* diopiGetVersion();

/**
 * operations to manipulate Tensor objects
 **/
extern DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t th, void**);
extern DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t th, const void**);
extern DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t th, diopiSize_t* size);
extern DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t th, diopiSize_t* stride);
extern DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t th, diopiDtype_t* dtype);
extern DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device);

extern DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel);
extern DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* itemsize);

/**
 * operations to require Stream and Tensor instances from a Context handle
 **/
extern DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream);

extern DIOPI_RT_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, const diopiSize_t* size, const diopiSize_t* stride,
                                                    const diopiDtype_t dtype, const diopiDevice_t device);
extern DIOPI_RT_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, int64_t num_bytes, diopiDevice_t device);

#if defined(__cplusplus)
}
#endif

#endif  // _PROJECT_DIOPERATOR_INTERFACE_RT_H_
