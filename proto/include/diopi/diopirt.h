/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_RT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_RT_H_

#include <stdint.h>

#define DIOPI_ATTR_WEAK __attribute__((weak))

#define DIOPI_API DIOPI_ATTR_WEAK
#define DIOPI_RT_API

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
    diopiForceFallbackToCPU = 14,
    diopiDtypeNotSupported = 1000,
} diopiError_t;

typedef enum {
    diopi_host = 0,
    diopi_device = 1,
} diopiDevice_t;

typedef int8_t diopiDeviceIndex_t;

// In order to meet the requirements of common dtype inference in diopi_test, all members in this enumeration should be assigned values in the order of
// increasing dtype precision
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

struct diopiGenerator;
typedef struct diopiGenerator* diopiGeneratorHandle_t;
typedef const struct diopiGenerator* diopiConstGeneratorHandle_t;

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
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device);

extern DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* itemsize);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorStoragePtr(diopiConstTensorHandle_t th, void** pStoragePtr);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorStorageOffset(diopiConstTensorHandle_t th, int64_t* pOffset);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorStorageNbytes(diopiConstTensorHandle_t th, size_t* pNbytes);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorDeviceIndex(diopiConstTensorHandle_t th, diopiDeviceIndex_t* pDevIndex);

extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorCrowIndices(diopiConstTensorHandle_t th, diopiConstTensorHandle_t* crow_indices);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorColIndices(diopiConstTensorHandle_t th, diopiConstTensorHandle_t* col_indices);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetTensorValues(diopiConstTensorHandle_t th, diopiConstTensorHandle_t* values);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiIsTensorSparse(diopiConstTensorHandle_t th, bool* is_sparse);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGetCurrentDeviceIndex(diopiDeviceIndex_t* pDevIndex);

/**
 * operations to require Stream and Tensor instances from a Context handle
 **/
extern DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream);

extern DIOPI_RT_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, const diopiSize_t* size, const diopiSize_t* stride,
                                                    const diopiDtype_t dtype, const diopiDevice_t device);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, int64_t num_bytes,
                                                                    diopiDevice_t device);

extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGeneratorGetState(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t th, diopiTensorHandle_t* data);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGeneratorSetState(diopiGeneratorHandle_t th, diopiConstTensorHandle_t state);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGeneratorGetSeedAndOffset(diopiGeneratorHandle_t th, uint64_t* ptrSeed, uint64_t* ptrOffset);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiGeneratorSetSeedAndOffset(diopiGeneratorHandle_t th, uint64_t seed, uint64_t offset);

/**
 * operations to manipulate profiler record objects.
 * Call diopiRecordStart at the beginning of code that you want to profile and call diopiRecordEnd at the end.
 **/
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiRecordStart(const char* record_name, void** record);
extern DIOPI_RT_API DIOPI_ATTR_WEAK diopiError_t diopiRecordEnd(void** record);

#if defined(__cplusplus)
}
#endif

#endif  // _PROJECT_DIOPERATOR_INTERFACE_RT_H_
