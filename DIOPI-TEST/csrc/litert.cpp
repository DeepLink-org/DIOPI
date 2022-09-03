/**
 * A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 *
**/

#include <diopi/diopirt.h>
#include <diopi_register.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <set>


extern "C" {

static int32_t DIOPIRT_LOG_LEVEL = 0;

#define PRINT_COLOR_NONE    "\033[0m"
#define PRINT_RED           "\033[1;31;40m"
#define PRINT_BLUE          "\033[1;34;40m"
#define PRINT_GREEN         "\033[1;32;40m"
#define PRINT_YELLOW        "\033[1;33;40m"

#define diopi_log(args...)                                                  \
    if (DIOPIRT_LOG_LEVEL) {                                                \
        fprintf(stdout, PRINT_BLUE " [ %s ] " PRINT_GREEN, __FUNCTION__);   \
        fprintf(stdout, args);                                              \
        fprintf(stdout, PRINT_COLOR_NONE "\n");                             \
    }

#define diopi_err(args)     fprintf(stderr, PRINT_RED args PRINT_COLOR_NONE)


static char szVersion[256] = {0};

DIOPI_API const char* diopiGetVersion()
{
    static bool inited = false;
    if (!inited) {
        inited = true;
        sprintf(szVersion, "DIOPI Version: %d.%d.%d", DIOPI_VER_MAJOR, DIOPI_VER_MINOR, DIOPI_VER_PATCH);
    }
    return szVersion;
}

static void* fake_device_malloc_func(uint64_t bytes)
{
    diopi_err("The device memory malloc function is not registered\n");
    exit(diopiNoRegisteredDeviceMemoryMallocFunction);
    return nullptr;
}

static malloc_func_t device_malloc_func = fake_device_malloc_func;
DIOPI_API diopiError_t diopiRegisterDeviceMallocFunc(malloc_func_t f)
{
    device_malloc_func = f;
    return diopiSuccess;
}

static void fake_device_memory_free_func(void* ptr)
{
    diopi_err("The device memory free function is not registered!\n");
    exit(diopiNoRegisteredDeviceMemoryFreeFunction);
}

static free_func_t device_mem_free_func = fake_device_memory_free_func;
DIOPI_API diopiError_t diopiRegisterDevMemFreeFunc(free_func_t f)
{
    device_mem_free_func = f;
    return diopiSuccess;
}

static void* host_malloc(uint64_t bytes)
{
    return malloc(bytes);
}

static void host_free(void* ptr)
{
    free(ptr);
}

static void* device_malloc(uint64_t bytes)
{
    void* ptr = device_malloc_func(bytes);
    diopi_log("device malloc bytes: %lu @ %16p", bytes, ptr);
    return ptr;
}

static void device_free(void* ptr)
{
    diopi_log("free device memory:%16p", ptr);
    device_mem_free_func(ptr);
}

static int32_t fake_memcpy_h2d_async_func(diopiStreamHandle_t stream, void* dst,
                                          const void* src, uint64_t bytes)
{
    diopi_err("The memcpy_h2d_async_func function is not registered, no operation is performed!\n");
    return diopiNoRegisteredHost2DeviceMemoryCopyFunction;
}

static memcpy_h2d_async_func_t memcpy_h2d_async_func = fake_memcpy_h2d_async_func;
DIOPI_API diopiError_t diopiRegisterMemcpyH2DAsyncFunc(memcpy_h2d_async_func_t f)
{
    diopi_log("memcpy_h2d_async_func_t:%16p", f);
    memcpy_h2d_async_func = f;
    return diopiSuccess;
}

static int32_t fake_memcpy_d2h_async_func(diopiStreamHandle_t stream, void* dst, const void* src,
                                          uint64_t bytes)
{
    diopi_err("The memcpy_d2h_async_func function is not registered, no operation is performed!\n");
    return diopiNoRegisteredDevice2HostMemoryCopyFunction;
}

static memcpy_d2h_async_func_t memcpy_d2h_async_func = fake_memcpy_d2h_async_func;
DIOPI_API diopiError_t diopiRegisterMemcpyD2HAsyncFunc(memcpy_d2h_async_func_t f)
{
    diopi_log("memcpy_d2h_async_func_t: %16p", f);
    memcpy_d2h_async_func = f;
    return diopiSuccess;
}

static int32_t fake_memcpy_d2d_async_func(diopiStreamHandle_t stream, void* dst, const void* src,
                                          uint64_t bytes)
{
    diopi_err("The memcpy_d2d_async_func function is not registered, no operation is performed!\n");
    return diopiNoRegisteredDevice2HostMemoryCopyFunction;
}

static memcpy_d2d_async_func_t memcpy_d2d_async_func = fake_memcpy_d2d_async_func;
DIOPI_API diopiError_t diopiRegisterMemcpyD2DAsyncFunc(memcpy_d2d_async_func_t f)
{
    memcpy_d2d_async_func = f;
    return diopiSuccess;
}

create_stream_func_t stream_create_func = nullptr;
DIOPI_API diopiError_t diopiRegisterStreamCreateFunc(create_stream_func_t f)
{
    stream_create_func = f;
    return diopiSuccess;
}

destroy_stream_func_t registered_stream_destroy_func = nullptr;
DIOPI_API diopiError_t diopiRegisterStreamDestroyFunc(destroy_stream_func_t f)
{
    registered_stream_destroy_func = f;
    return diopiSuccess;
}

static int32_t fake_synchronize_stream(diopiStreamHandle_t stream)
{
    diopi_err("The stream synchronization function is not registered!\n");
    return diopiNoRegisteredStreamSyncFunction;
}

static sync_stream_func_t synchronize_stream_func = fake_synchronize_stream;
DIOPI_API diopiError_t diopiRegisterSynchronizeStreamFunc(sync_stream_func_t f)
{
    synchronize_stream_func = f;
    return diopiSuccess;
}

static const char* fake_get_last_error_string_func()
{
    diopi_err("The get_last_error_string function is not registered.\n");
    return nullptr;
}

static get_last_error_string_func_t get_last_error_string_func = fake_get_last_error_string_func;
DIOPI_API diopiError_t diopiRegisterGetLastErrorFunc(get_last_error_string_func_t f)
{
    get_last_error_string_func = f;
    return diopiSuccess;
}

DIOPI_API const char* diopiGetLastErrorString() {
    return get_last_error_string_func();
}

void _getLastErrorString(const char** strErr) {
    const char* str = diopiGetLastErrorString();
    *strErr = str;
}

int32_t itemsize(const diopiDtype_t dtype)
{
    switch (dtype)
    {
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

const char* diopi_dtype_to_str(const diopiDtype_t dtype)
{
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

const char* device_to_str(const diopiDevice_t device)
{
#define _device2str(type) \
    if (type == device) return #type;
    _device2str(diopi_host);
    _device2str(diopi_device);

    return "Unknown device type\n";
#undef _device2str
}

class Storage final {
private:
    malloc_func_t malloc_fn_;
    free_func_t   free_fn_;
    int64_t       nbytes_;
    void*         ptr_;

public:
    Storage(malloc_func_t malloc_fn, free_func_t free_fn, int64_t nbytes)
            : malloc_fn_(malloc_fn), free_fn_(free_fn), nbytes_(nbytes) {
        assert(free_fn_);
        assert(malloc_fn_);
        ptr_ = malloc_fn_(nbytes_);
    }

    ~Storage() {
        free_fn_(ptr_);
        ptr_    = nullptr;
        nbytes_ = 0;
    }

    void*       data() { return ptr_; }
    const void* data() const { return ptr_; }
    int64_t     nbytes() const { return nbytes_; }
};

struct diopiTensor {
private:
    std::vector<int64_t>     shape_;
    std::vector<int64_t>     stride_;
    diopiDtype_t             dtype_;
    diopiDevice_t            device_;
    int64_t                  numel_;
    std::shared_ptr<Storage> storage_;

public:
    diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride,
                const diopiDtype_t dtype, const diopiDevice_t device);
    ~diopiTensor();

    diopiSize_t shape() {
        diopiSize_t size(shape_.data(), static_cast<int64_t>(shape_.size()));
        return size;
    }

    diopiSize_t stride() {
        diopiSize_t stride(stride_.data(), static_cast<int64_t>(stride_.size()));
        return stride;
    }

    bool reset_shape(const diopiSize_t* size);

    diopiDtype_t  dtype() const { return dtype_; }
    diopiDevice_t device() const { return device_; }
    int64_t       numel() const { return numel_; }

    void*       data() { return storage_->data(); }
    const void* data() const { return storage_->data(); }
    int64_t     nbytes() const { return storage_->nbytes(); }

};

diopiTensor::diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride,
                         const diopiDtype_t dtype, const diopiDevice_t device) {
    assert(shape);
    dtype_  = dtype;
    device_ = device;

    shape_.resize(shape->len);
    stride_.resize(shape->len);
    int64_t stride_temp = 1;
    numel_              = 1;
    for (int64_t i = shape->len - 1; i >= 0; --i) {
        shape_[i] = shape->data[i];
        numel_ *= shape->data[i];
        if (stride != nullptr) {
            stride_[i] = stride->data[i];
        } else {
            stride_[i] = stride_temp;
            stride_temp *= shape->data[i];
        }
    }

    const int64_t nbytes = numel_ * itemsize(dtype_);
    if (device_ == diopi_host) {
        storage_ = std::make_shared<Storage>(host_malloc, host_free, nbytes);
    } else {
        storage_ = std::make_shared<Storage>(device_malloc, device_free, nbytes);
    }
}

bool diopiTensor::reset_shape(const diopiSize_t* size) {
    int64_t numel = 1;
    for (int64_t i = 0; i < size->len; ++i) {
        numel *= size->data[i];
    }
    if (numel != numel_) return false;

    shape_.resize(size->len);
    stride_.resize(size->len);
    int64_t stride_temp = 1;
    for (int64_t i = size->len - 1; i >= 0; --i) {
        shape_[i] = size->data[i];
        stride_[i] = stride_temp;
        stride_temp *= size->data[i];
    }
    return true;
}

diopiTensor::~diopiTensor() {}

DIOPI_API diopiError_t diopiGetTensorData(diopiTensorHandle_t* pth, void** pptr) {
    *pptr = (*pth)->data();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDataConst(const diopiTensorHandle_t* pth, const void** pptr) {
    *pptr = (*pth)->data();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorShape(const diopiTensorHandle_t th, diopiSize_t* size) {
    *size = th->shape();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorStride(const diopiTensorHandle_t th, diopiSize_t* stride) {
    *stride = th->stride();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDtype(const diopiTensorHandle_t th, diopiDtype_t* dtype) {
    *dtype = th->dtype();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDevice(const diopiTensorHandle_t th, diopiDevice_t* device) {
    *device = th->device();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorNumel(const diopiTensorHandle_t th, int64_t* numel) {
    *numel = th->numel();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorElemSize(const diopiTensorHandle_t th, int64_t* elem_size) {
    *elem_size = itemsize(th->dtype());
    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiTensorResetShape(const diopiTensorHandle_t th, const diopiSize_t* size) {
    if (!th->reset_shape(size)) {
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}


struct diopiContext {
private:
    diopiStreamHandle_t stream_ { nullptr };
    std::set<diopiTensorHandle_t> setTensors_;

public:
    diopiContext() {}

    ~diopiContext() {
        if (nullptr != stream_) {
            if (registered_stream_destroy_func != nullptr) {
                registered_stream_destroy_func(stream_);
            }
        }
        for (auto it : setTensors_) {
            delete it;
        }
        setTensors_.clear();
    }

    diopiStreamHandle_t getStreamHandle() {
        if (stream_ == nullptr) {
            if (stream_create_func == nullptr) {
                diopi_err("stream create function is not registered!\n");
            } else {
                stream_create_func(&stream_);
            }
        }
        return stream_;
    }

    diopiTensorHandle_t createTensor(const diopiSize_t* size, const diopiSize_t* stride,
                                     const diopiDtype_t dtype, const diopiDevice_t dev) {
        diopiTensorHandle_t tensor = new diopiTensor(size, stride, dtype, dev);
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
};

DIOPI_API diopiError_t _diopiCreateContext(diopiContextHandle_t* ctx) {
    *ctx = new diopiContext();
    diopi_log("create a Context instance: %16p", *ctx);
    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiDestroyContext(diopiContextHandle_t ctx) {
    diopi_log("destroy a Context instance: %16p", ctx);
    delete ctx;
    return diopiSuccess;
}

diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    if (nullptr == stream_create_func) {
        diopi_err("stream create function is not registered!\n");
        return diopiNoRegisteredStreamCreateFunction;
    }
    *stream = ctx->getStreamHandle();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          const diopiSize_t* size, const diopiSize_t* stride,
                                          const diopiDtype_t dtype, const diopiDevice_t dev) {
    diopi_log("requires a Tensor, size:[%16p, %lld], stride:%16p, dtype:%d[%s], device:%d[%s]",
        size->data, size->len, stride, dtype, diopi_dtype_to_str(dtype), dev, device_to_str(dev));
    *tensor = ctx->createTensor(size, stride, dtype, dev);

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRequireBuffer(diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
                                          int64_t bytes, diopiDevice_t dev) {
    diopi_log("requires a buffer, bytes: %lld, device: %s", bytes, device_to_str(dev));
    diopiSize_t size(&bytes, 1);
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, dev);
}

DIOPI_API diopiError_t _diopiDestoryTensor(diopiContextHandle_t ctx, diopiTensorHandle_t tensor) {
    ctx->destroyTensor(tensor);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiInit() {
    static int32_t inited = 0;
    if (inited) {
        return diopiSuccess;
    }
    inited = 1;
    const char* log_level_env = getenv("DIOPIRT_LOG_LEVEL");
    if (log_level_env != nullptr) {
        DIOPIRT_LOG_LEVEL = atoi(log_level_env);
    } else {
        DIOPIRT_LOG_LEVEL = 0;
    }
    diopi_log("DIOPIRT_LOG_LEVEL:%d", DIOPIRT_LOG_LEVEL);

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiFinalize() {
    static int32_t finalized = 0;
    if (finalized) {
        return diopiSuccess;
    }
    finalized = 1;

    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiTensorCopyFromBuffer(diopiContextHandle_t  ctx,
                                                  const void*           src,
                                                  diopiTensorHandle_t   tensor) {
    if (tensor->device() == diopi_device) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        memcpy_h2d_async_func(stream, tensor->data(), src, tensor->nbytes());
        synchronize_stream_func(stream);
    } else {
        std::memcpy(tensor->data(), src, tensor->nbytes());
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiTensorCopyToBuffer(diopiContextHandle_t        ctx,
                                                const diopiTensorHandle_t   tensor,
                                                void*                       dst) {
    if (tensor->device() == diopi_device) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        memcpy_d2h_async_func(stream, dst, tensor->data(), tensor->nbytes());
        synchronize_stream_func(stream);
    } else {
        std::memcpy(dst, tensor->data(), tensor->nbytes());
    }
    return diopiSuccess;
}

}  // extern "C"
