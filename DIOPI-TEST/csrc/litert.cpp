/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink Inc.
 * @brief A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 */

#include <conform_test.h>
#include <diopi/diopirt.h>
#include <inttypes.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

extern "C" {

static int32_t DIOPIRT_LOG_LEVEL = 0;

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

class Storage final {
private:
    malloc_func_t mallocFn_;
    free_func_t freeFn_;
    int64_t nbytes_;
    void* ptr_;

public:
    Storage(malloc_func_t mallocFn, free_func_t freeFn, int64_t nbytes) : mallocFn_(mallocFn), freeFn_(freeFn), nbytes_(nbytes) {
        assert(freeFn);
        assert(mallocFn);
        ptr_ = mallocFn(nbytes);
    }

    ~Storage() {
        freeFn_(ptr_);
        ptr_ = nullptr;
        nbytes_ = 0;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    int64_t nbytes() const { return nbytes_; }
};

struct diopiTensor {
private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    diopiDtype_t dtype_;
    diopiDevice_t device_;
    int64_t numel_;
    std::shared_ptr<Storage> storage_;
    diopiContextHandle_t context_;

public:
    diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride, diopiDtype_t dtype, diopiDevice_t device, diopiContextHandle_t context);
    ~diopiTensor();

    diopiSize_t shape() const {
        diopiSize_t size(shape_.data(), static_cast<int64_t>(shape_.size()));
        return size;
    }

    diopiSize_t stride() const {
        diopiSize_t stride(stride_.data(), static_cast<int64_t>(stride_.size()));
        return stride;
    }

    bool resetShape(const diopiSize_t* size);

    diopiDtype_t dtype() const { return dtype_; }
    diopiDevice_t device() const { return device_; }
    int64_t numel() const { return numel_; }

    void* data() { return storage_->data(); }
    const void* data() const { return storage_->data(); }
    int64_t nbytes() const { return storage_->nbytes(); }

    diopiContextHandle_t getCtx() const { return context_; }
};

diopiTensor::diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride, diopiDtype_t dtype, diopiDevice_t device, diopiContextHandle_t context) {
    assert(shape);
    dtype_ = dtype;
    device_ = device;

    shape_.resize(shape->len);
    stride_.resize(shape->len);
    int64_t strideTemp = 1;
    numel_ = 1;
    for (int64_t i = shape->len - 1; i >= 0; --i) {
        shape_[i] = shape->data[i];
        numel_ *= shape->data[i];
        if (stride != nullptr) {
            stride_[i] = stride->data[i];
        } else {
            stride_[i] = strideTemp;
            strideTemp *= shape->data[i];
        }
    }

    const int64_t nbytes = numel_ * itemsize(dtype);
    if (device == diopi_host) {
        storage_ = std::make_shared<Storage>(hostMalloc, hostFree, nbytes);
    } else {
        storage_ = std::make_shared<Storage>(device_malloc, device_free, nbytes);
    }
    context = context;
}

bool diopiTensor::resetShape(const diopiSize_t* size) {
    int64_t numel = 1;
    for (int64_t i = 0; i < size->len; ++i) {
        numel *= size->data[i];
    }
    if (numel != numel) return false;

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

diopiTensor::~diopiTensor() {}

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

DIOPI_RT_API diopiError_t diopiTensorResetShape(diopiTensorHandle_t th, const diopiSize_t* size) {
    if (!th->resetShape(size)) {
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiTensorGetCtxHandle(diopiConstTensorHandle_t th, diopiContextHandle_t* ctx) {
    *ctx = th->getCtx();
    return diopiSuccess;
}

struct diopiContext {
private:
    diopiStreamHandle_t stream_{nullptr};
    std::set<diopiTensorHandle_t> setTensors_;

public:
    diopiContext() {}

    ~diopiContext() {
        if (nullptr != stream_) {
            device_destroy_stream(stream_);
        }
        for (auto it : setTensors_) {
            delete it;
        }
        setTensors_.clear();
    }

    diopiStreamHandle_t getStreamHandle() {
        if (stream_ == nullptr) {
            device_make_stream(&stream_);
        }
        return stream_;
    }

    diopiTensorHandle_t createTensor(const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype, const diopiDevice_t dev) {
        diopiTensorHandle_t tensor = new diopiTensor(size, stride, dtype, dev, this);
        setTensors_.insert(tensor);
        return tensor;
    }

    void destroyTensor(diopiTensorHandle_t tensor) {
        auto it = setTensors_.find(tensor);
        if (setTensors_.end() != it) {
            setTensors_.erase(it);
            delete tensor;
        }
    }

    void clearTensors() {
        if (stream_ != nullptr) {
            for (auto it : setTensors_) {
                delete it;
            }
            setTensors_.clear();
        }
    }
};

DIOPI_RT_API diopiError_t diopiCreateContext(diopiContextHandle_t* ctx) {
    *ctx = new diopiContext();
    diopi_log("create a Context instance: %16p", *ctx);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiDestroyContext(diopiContextHandle_t ctx) {
    diopi_log("destroy a Context instance: %16p", ctx);
    delete ctx;
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
    diopiSize_t size(&bytes, 1);
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, dev);
}

DIOPI_RT_API diopiError_t diopiDestoryTensor(diopiContextHandle_t ctx, diopiTensorHandle_t tensor) {
    ctx->destroyTensor(tensor);
    return diopiSuccess;
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

DIOPI_RT_API diopiError_t diopiClearTensors(diopiContextHandle_t ctx) {
    ctx->clearTensors();
    return diopiSuccess;
}

}  // extern "C"
