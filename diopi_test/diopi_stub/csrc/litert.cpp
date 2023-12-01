/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink Inc.
 * @brief A reference implemention for DIOPI runtime, which is utilized to
 * support conformance test suite of DIOPI
 */

#define __STDC_FORMAT_MACROS
#include "litert.hpp"

#include <conform_test.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

extern "C" {

static int32_t DIOPIRT_LOG_LEVEL = 1;

#define PRINT_COLOR_NONE "\033[0m"
#define PRINT_RED "\033[1;31;40m"
#define PRINT_BLUE "\033[1;34;40m"
#define PRINT_GREEN "\033[1;32;40m"
#define PRINT_YELLOW "\033[1;33;40m"

#define diopi_log(args...)                                                \
    if (DIOPIRT_LOG_LEVEL) {                                              \
        fprintf(stdout, PRINT_BLUE " [ %s ] " PRINT_GREEN, __FUNCTION__); \
        fprintf(stdout, args);                                            \
        fprintf(stdout, PRINT_COLOR_NONE "\n");                           \
    }

#define diopi_err(args) fprintf(stderr, PRINT_RED args PRINT_COLOR_NONE)

static char szVersion[256] = {0};

DIOPI_RT_API const char* diopiGetVersion() {
    static bool inited = false;
    if (!inited) {
        inited = true;
        sprintf(szVersion, "DIOPI Version: %d.%d.%d", DIOPI_VER_MAJOR, DIOPI_VER_MINOR, DIOPI_VER_PATCH);
    }
    return szVersion;
}

static void* hostMalloc(uint64_t bytes) { return malloc(bytes); }

static void hostFree(void* ptr) { free(ptr); }

int32_t itemsize(const diopiDtype_t dtype) {
    switch (dtype) {
        case diopi_dtype_int32:
        case diopi_dtype_uint32:
        case diopi_dtype_float32:
        case diopi_dtype_tfloat32:
            return 4;
        case diopi_dtype_int64:
        case diopi_dtype_uint64:
        case diopi_dtype_float64:
        case diopi_dtype_complex64:
            return 8;
        case diopi_dtype_int16:
        case diopi_dtype_uint16:
        case diopi_dtype_float16:
        case diopi_dtype_bfloat16:
            return 2;
        case diopi_dtype_int8:
        case diopi_dtype_uint8:
        case diopi_dtype_bool:
            return 1;
        case diopi_dtype_complex128:
            return 16;
        default:
            assert(0);
    }
    return 0;
}

const char* diopiDtypeToStr(const diopiDtype_t dtype) {
#define _dtype2str(type) \
    if (type == dtype) return #type;
    _dtype2str(diopi_dtype_float16);
    _dtype2str(diopi_dtype_float32);
    _dtype2str(diopi_dtype_float64);
    _dtype2str(diopi_dtype_int8);
    _dtype2str(diopi_dtype_uint8);
    _dtype2str(diopi_dtype_int16);
    _dtype2str(diopi_dtype_uint16);
    _dtype2str(diopi_dtype_int32);
    _dtype2str(diopi_dtype_uint32);
    _dtype2str(diopi_dtype_int64);
    _dtype2str(diopi_dtype_uint64);
    _dtype2str(diopi_dtype_bool);
    _dtype2str(diopi_dtype_bfloat16);
    _dtype2str(diopi_dtype_tfloat32);
    _dtype2str(diopi_dtype_complex64);
    _dtype2str(diopi_dtype_complex128);

    return nullptr;
#undef _dtype2str
}

const char* deviceToStr(const diopiDevice_t device) {
#define _device2str(type) \
    if (type == device) return #type;
    _device2str(diopi_host);
    _device2str(diopi_device);

    return "Unknown device type\n";
#undef _device2str
}

diopiTensor::diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride, diopiDtype_t dtype, diopiDevice_t device, diopiContextHandle_t context,
                         const void* src)
    : dtype_(dtype), device_(device), context_(context) {
    assert(shape);

    shape_.resize(shape->len);
    stride_.resize(shape->len);
    int64_t strideTemp = 1;
    numel_ = 1;
    int64_t strideNumel = 1;
    for (int64_t i = shape->len - 1; i >= 0; --i) {
        shape_[i] = shape->data[i];
        numel_ *= shape->data[i];
        if (stride != nullptr && stride->data != nullptr) {
            stride_[i] = stride->data[i];
        } else {
            stride_[i] = strideTemp;
            strideTemp *= shape->data[i];
        }
        strideNumel += (shape_[i] - 1) * stride_[i];
    }
    strideNumel *= itemsize(dtype);
    const int64_t nbytes = strideNumel;
    // const int64_t nbytes = numel_ * itemsize(dtype);
    if (device_ == diopi_host) {
        storage_ = std::make_shared<Storage>(hostMalloc, hostFree, nbytes);
    } else {
        storage_ = std::make_shared<Storage>(device_malloc, device_free, nbytes);
    }
    if (src != nullptr) {
        diopiTensorCopyFromBuffer(context, src, this);
    }
}

diopiTensor& diopiTensor::operator=(const diopiTensor& other) {
    if (this == &other) {
        return *this;
    }

    shape_ = other.shape_;
    stride_ = other.stride_;
    dtype_ = other.dtype_;
    device_ = other.device_;
    numel_ = other.numel_;
    context_ = other.context_;
    if (device_ == diopi_host) {
        storage_ = std::make_shared<Storage>(hostMalloc, hostFree, other.nbytes());
    } else {
        storage_ = std::make_shared<Storage>(device_malloc, device_free, other.nbytes());
    }

    const void* src = other.data();
    if (src == nullptr) {
        return *this;
    }

    if (device_ == diopi_host) {
        std::memcpy(data(), src, other.nbytes());
    } else {
        diopiStreamHandle_t stream;
        diopiGetStream(context_, &stream);
        device_memcpy_d2d_async(stream, data(), src, other.nbytes());
        device_synchronize_stream(stream);
    }
    return *this;
}

bool diopiTensor::resetShape(const diopiSize_t* size) {
    int64_t numel = 1;
    for (int64_t i = 0; i < size->len; ++i) {
        numel *= size->data[i];
    }
    if (numel != numel_) return false;

    shape_.resize(size->len);
    stride_.resize(size->len);
    int64_t strideTemp = 1;
    for (int64_t i = size->len - 1; i >= 0; --i) {
        shape_[i] = size->data[i];
        stride_[i] = strideTemp;
        strideTemp *= size->data[i];
    }
    return true;
}

DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t th, void** pptr) {
    *pptr = th->data();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t th, const void** pptr) {
    *pptr = th->data();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t th, diopiSize_t* size) {
    *size = th->shape();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t th, diopiSize_t* stride) {
    *stride = th->stride();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t th, diopiDtype_t* dtype) {
    *dtype = th->dtype();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t th, diopiDevice_t* device) {
    *device = th->device();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t th, int64_t* numel) {
    *numel = th->numel();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t th, int64_t* elemSize) {
    *elemSize = itemsize(th->dtype());
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStoragePtr(diopiConstTensorHandle_t th, void** pStoragePtr) {
    *pStoragePtr = const_cast<void*>(th->data());
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStorageOffset(diopiConstTensorHandle_t th, int64_t* pOffset) {
    *pOffset = 0;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStorageNbytes(diopiConstTensorHandle_t th, size_t* pNbytes) {
    *pNbytes = th->nbytes();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDeviceIndex(diopiConstTensorHandle_t th, diopiDeviceIndex_t* pDevIndex) {
    *pDevIndex = 0;
    return diopiSuccess;
}

diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    *stream = ctx->getStreamHandle();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, const diopiSize_t* size, const diopiSize_t* stride,
                                             const diopiDtype_t dtype, const diopiDevice_t dev) {
    diopi_log("requires a Tensor, size:[%16p, %" PRId64 "], stride:%16p, dtype:%d[%s], device:%d[%s]",
              size->data,
              size->len,
              stride,
              dtype,
              diopiDtypeToStr(dtype),
              dev,
              deviceToStr(dev));
    *tensor = ctx->createTensor(size, stride, dtype, dev);

    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor, int64_t bytes, diopiDevice_t dev) {
    diopi_log("requires a buffer, bytes: %" PRId64 ", device: %s", bytes, deviceToStr(dev));
    diopiSize_t size{&bytes, 1};
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, dev);
}

DIOPI_RT_API diopiError_t diopiInit() {
    static int32_t inited = 0;
    if (inited) {
        return diopiSuccess;
    }
    inited = 1;
    const char* logLevelEnv = getenv("DIOPIRT_LOG_LEVEL");
    if (logLevelEnv != nullptr) {
        DIOPIRT_LOG_LEVEL = atoi(logLevelEnv);
    } else {
        DIOPIRT_LOG_LEVEL = 0;
    }
    diopi_log("DIOPIRT_LOG_LEVEL:%d", DIOPIRT_LOG_LEVEL);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiFinalize() {
    static int32_t finalized = 0;
    if (finalized) {
        return diopiSuccess;
    }
    finalized = 1;

    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiTensorCopyFromBuffer(diopiContextHandle_t ctx, const void* src, diopiTensorHandle_t tensor) {
    if (tensor->device() == diopi_device) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        device_memcpy_h2d_async(stream, tensor->data(), src, tensor->nbytes());
        device_synchronize_stream(stream);
    } else {
        std::memcpy(tensor->data(), src, tensor->nbytes());
    }
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiTensorCopyToBuffer(diopiContextHandle_t ctx, diopiConstTensorHandle_t tensor, void* dst) {
    if (tensor->device() == diopi_device) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        device_memcpy_d2h_async(stream, dst, tensor->data(), tensor->nbytes());
        device_synchronize_stream(stream);
    } else {
        std::memcpy(dst, tensor->data(), tensor->nbytes());
    }
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorGetState(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t th, diopiTensorHandle_t* data) {
    const diopiTensor& state = th->state();
    diopiDtype_t dtype;
    diopiGetTensorDtype(&state, &dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(&state, &device);
    diopiSize_t shape;
    diopiGetTensorShape(&state, &shape);
    diopiSize_t stride;
    diopiGetTensorStride(&state, &stride);
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &shape, &stride, dtype, diopi_device);
    *tensor = state;
    *data = tensor;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorSetState(diopiGeneratorHandle_t th, diopiConstTensorHandle_t state) {
    th->set_state(state);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordStart(const char* recordName, void** record) { return diopiSuccess; }

DIOPI_RT_API diopiError_t diopiRecordEnd(void** record) { return diopiSuccess; }

}  // extern "C"
