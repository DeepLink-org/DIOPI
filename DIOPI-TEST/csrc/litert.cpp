/**
 * A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 *
**/

#include <diopi/diopirt.h>
#include <diopi_register.h>

#include <assert.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
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


static void* fake_device_malloc_fun(uint64_t bytes)
{
    diopi_err("The device memory malloc function is not registered\n");
    exit(diopiNoRegisteredDeviceMemoryMallocFunction);
    return NULL;
}

static malloc_func_t device_malloc_func = fake_device_malloc_fun;
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
    diopi_err("The memcpy_h2d_async_fun function is not registered, no operation is performed!\n");
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
    diopi_err("The memcpy_d2h_async_fun function is not registered, no operation is performed!\n");
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
    diopi_err("The memcpy_d2d_async_fun function is not registered, no operation is performed!\n");
    return diopiNoRegisteredDevice2HostMemoryCopyFunction;
}

static memcpy_d2d_async_func_t memcpy_d2d_async_func = fake_memcpy_d2d_async_func;
DIOPI_API diopiError_t diopiRegisterMemcpyD2DAsyncFunc(memcpy_d2d_async_func_t f)
{
    memcpy_d2d_async_func = f;
    return diopiSuccess;
}

create_stream_func_t stream_create_func = NULL;
DIOPI_API diopiError_t diopiRegisterStreamCreateFunc(create_stream_func_t f)
{
    stream_create_func = f;
    return diopiSuccess;
}

destroy_stream_func_t registered_stream_destroy_fun = NULL;
DIOPI_API diopiError_t diopiRegisterStreamDestroyFunc(destroy_stream_func_t f)
{
    registered_stream_destroy_fun = f;
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

static const char* fake_get_last_error_string_fun()
{
    diopi_err("The get_last_error_string function is not registered.\n");
    return NULL;
}

static get_last_error_string_func_t get_last_error_string_func = fake_get_last_error_string_fun;
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
    if ((dtype == diopi_dtype_int32) || (dtype == diopi_dtype_uint32) ||
        (dtype == diopi_dtype_float32) || (dtype == diopi_dtype_tfloat32)) {
        return 4;
    } else if ((dtype == diopi_dtype_int64) || (dtype == diopi_dtype_uint64) ||
             (dtype == diopi_dtype_float64)) {
        return 8;
    } else if ((dtype == diopi_dtype_int16) || (dtype == diopi_dtype_uint16) ||
             (dtype == diopi_dtype_float16) || (dtype == diopi_dtype_bfloat16)) {
        return 2;
    } else if ((dtype == diopi_dtype_int8) || (dtype == diopi_dtype_uint8) ||
             (dtype == diopi_dtype_bool)) {
        return 1;
    } else {
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

    return NULL;
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
    malloc_func_t malloc_fun_;
    free_func_t   free_fun_;
    int64_t       nbytes_;
    void*         ptr_;

public:
    Storage(malloc_func_t malloc_fun, free_func_t free_fun, int64_t nbytes)
        : malloc_fun_(malloc_fun)
        , free_fun_(free_fun)
        , nbytes_(nbytes) {
        assert(free_fun_);
        assert(malloc_fun_);
        ptr_ = malloc_fun_(nbytes_);
    }

    ~Storage() {
        free_fun_(ptr_);
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
    uint64_t                 id_;
    std::shared_ptr<Storage> storage_ptr_;

public:
    diopiTensor(const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype,
                const diopiDevice_t device);
    ~diopiTensor();

    diopiSize_t stride() {
        diopiSize_t size(stride_.data(), stride_.size());
        return size;
    }

    diopiSize_t shape() {
        diopiSize_t size(shape_.data(), shape_.size());
        return size;
    }

    diopiDtype_t  dtype() const { return dtype_; }
    diopiDevice_t device() const { return device_; }
    int64_t       numel() const { return numel_; }

    void*       data() { return storage_ptr_->data(); }
    const void* data() const { return storage_ptr_->data(); }
    int64_t     nbytes() const { return storage_ptr_->nbytes(); }
};

diopiTensor::diopiTensor(const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype,
                         const diopiDevice_t device) {
    assert(size);
    const int64_t shape_len = size->len;
    if (shape_len <= 0) {
        return;
    }
    dtype_  = dtype;
    device_ = device;

    shape_.resize(shape_len);
    stride_.resize(shape_len);
    int64_t stride_temp = 1;
    numel_              = 1;
    for (int64_t i = shape_len - 1; i >= 0; i--) {
        shape_[i] = size->data[i];
        numel_ *= size->data[i];
        if (stride != nullptr) {
            stride_[i] = stride->data[i];
        }
        else {
            stride_[i] = stride_temp;
            stride_temp *= size->data[i];
        }
    }

    const uint64_t nbytes = numel_ * itemsize(dtype_);
    if (device_ == diopi_host) {
        storage_ptr_ = std::make_shared<Storage>(host_malloc, host_free, nbytes);
    }
    else {
        storage_ptr_ = std::make_shared<Storage>(device_malloc, device_free, nbytes);
    }
}

diopiTensor::~diopiTensor() {}

DIOPI_API diopiError_t diopiGetTensorData(diopiTensorHandle_t* th, void** pptr) {
    *pptr = (*th)->data();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiGetTensorDataConst(const diopiTensorHandle_t* th, const void** pptr) {
    *pptr = (*th)->data();
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


struct diopiContext {
private:
    diopiStreamHandle_t stream_ = nullptr;
    std::set<diopiTensorHandle_t> setTensors_;

public:
    diopiContext() {}

    ~diopiContext() {
        if (stream_ != nullptr) {
            if (registered_stream_destroy_fun != nullptr) {
                registered_stream_destroy_fun(stream_);
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

DIOPI_API diopiError_t diopiCreateContext(diopiContextHandle_t* ctx) {
    *ctx = new diopiContext();
    diopi_log("create a Context instance: %16p", *ctx);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDestoryContext(diopiContextHandle_t ctx) {
    diopi_log("destroy a Context instance: %16p", ctx);
    delete ctx;
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
    return diopiRequireTensor(ctx, tensor, &size, NULL, diopi_dtype_int8, dev);
}

DIOPI_API diopiError_t diopiDestoryTensor(diopiContextHandle_t ctx, diopiTensorHandle_t tensor) {
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
    if (log_level_env != NULL) {
        DIOPIRT_LOG_LEVEL = atoi(log_level_env);
    }
    else {
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

diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    if (stream_create_func == NULL) {
        diopi_err("stream create function is not registered!\n");
        return diopiNoRegisteredStreamCreateFunction;
    }
    *stream = ctx->getStreamHandle();
    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiTensorCopyHostToDevice(diopiContextHandle_t      ctx,
                                                    const diopiTensorHandle_t cpu_tensor,
                                                    diopiTensorHandle_t       device_tensor) {
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    memcpy_h2d_async_func(stream, device_tensor->data(), cpu_tensor->data(), cpu_tensor->nbytes());
    synchronize_stream_func(stream);
    return diopiSuccess;
}

DIOPI_API diopiError_t _diopiTensorCopyDeviceToHost(diopiContextHandle_t      ctx,
                                                    const diopiTensorHandle_t device_tensor,
                                                    diopiTensorHandle_t       cpu_tensor) {
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    memcpy_d2h_async_func(stream, cpu_tensor->data(), device_tensor->data(), device_tensor->nbytes());
    synchronize_stream_func(stream);
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

void print_tensor_elem(diopiTensorHandle_t tensor) {
    int64_t elemnu;
    diopiGetTensorNumel(tensor, &elemnu);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);

    const size_t shape_len = shape.len;
    diopiSize_t  pos;
    pos.len = shape_len;
    int64_t array_buffer[shape_len];
    pos.data = array_buffer;

    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    int64_t     elemindex = 0;
    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);

    for (elemindex = 0; elemindex < elemnu; ++elemindex) {
        size_t  tempIndex = elemindex;
        int     dim       = 0;
        int64_t index     = 0;
        for (dim = shape_len - 1; dim >= 0; --dim) {
            pos.data[dim] = (tempIndex / stride.data[dim]) % shape.data[dim];
            index += pos.data[dim] * stride.data[dim];
        }
        size_t brackets = 0;
        for (size_t i = 0; i < shape_len; i++) {
            size_t j = 0;
            for (j = i; j < shape_len; j++) {
                if (pos.data[j] != 0) {
                    break;
                }
                if (j == shape_len - 1) {
                    brackets++;
                }
            }
        }
        if (brackets < shape_len) {
            printf("\t");
        }
        for (size_t i = 1; i < shape_len - brackets; i++) {
            printf(" ");
        }
        for (size_t i = 0; i < brackets; i++) {
            printf("[");
        }

        const void* tensor_data;
        diopiGetTensorDataConst(&tensor, &tensor_data);
        if (dtype == diopi_dtype_int32) {
            const int32_t* data = (int32_t*)tensor_data;
            printf("%-8d", data[index]);
        } else if (dtype == diopi_dtype_uint32) {
            const uint32_t* data = (uint32_t*)tensor_data;
            printf("%-8u", data[index]);
        }
        if (dtype == diopi_dtype_int16) {
            const int16_t* data = (int16_t*)tensor_data;
            printf("%-8d", (int32_t)data[index]);
        } else if (dtype == diopi_dtype_uint16) {
            const uint16_t* data = (uint16_t*)tensor_data;
            printf("%-8d", (int32_t)data[index]);
        }
        if (dtype == diopi_dtype_int8) {
            const int8_t* data = (int8_t*)tensor_data;
            printf("%-8d", (int32_t)data[index]);
        } else if (dtype == diopi_dtype_uint8) {
            const uint8_t* data = (uint8_t*)tensor_data;
            printf("%-8d", (int32_t)data[index]);
        } else if (dtype == diopi_dtype_float32) {
            const float* data = (float*)tensor_data;
            printf("%-6.6f", data[index]);
        } else if (dtype == diopi_dtype_float64) {
            const double* data = (double*)tensor_data;
            printf("%-12.8lf", data[index]);
        } else if (dtype == diopi_dtype_bool) {
            const char* data = (char*)tensor_data;
            printf("%s", data[index] ? "true" : "false");
        }

        if (pos.data[shape_len - 1] != (shape.data[shape_len - 1]) - 1) {
            printf(",");
        }

        brackets = 0;
        for (size_t i = 0; i < shape_len; i++) {
            for (size_t j = i; j < shape_len; j++) {
                if (pos.data[j] != shape.data[j] - 1) {
                    break;
                }
                if (j == shape_len - 1) {
                    brackets++;
                }
            }
        }
        for (size_t i = 0; i < brackets; i++) {
            printf("]");
            if (i == brackets - 1) {
                if (index <= elemnu - shape.data[shape_len - 1]) {
                    printf(",\n");
                    if (brackets >= 2) {
                        printf("\n");
                    }
                }
            }
        }
    }
}

DIOPI_API diopiError_t diopiDumpTensor(diopiContextHandle_t ctx, const diopiTensorHandle_t tensor) {
    diopi_log("ctx:%16p, tensor:%16p", ctx, tensor);
    int32_t DIOPIRT_LOG_LEVEL_TEMP = DIOPIRT_LOG_LEVEL;
    DIOPIRT_LOG_LEVEL              = 0;
    diopiSize_t   shape;
    diopiSize_t   stride;
    int64_t       numel;
    int64_t       elemsize;
    diopiDtype_t  dtype;
    diopiDevice_t device;
    const void*   data_ptr;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    diopiGetTensorNumel(tensor, &numel);
    diopiGetTensorElemSize(tensor, &elemsize);
    diopiGetTensorDtype(tensor, &dtype);
    diopiGetTensorDevice(tensor, &device);
    diopiGetTensorDataConst(&tensor, &data_ptr);
    if ((dtype == diopi_dtype_bfloat16) || (dtype == diopi_dtype_float16) ||
        (dtype == diopi_dtype_tfloat32)) {
        return diopiDtypeNotSupported;
    }
    const int64_t nbytes = numel * elemsize;
    printf("tensor(");
    if (diopi_host == device) {
        print_tensor_elem(tensor);
    } else {
        diopiTensorHandle_t cpu_tensor;
        diopiRequireTensor(ctx, &cpu_tensor, &shape, &stride, dtype, diopi_host);
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        memcpy_d2h_async_func(stream, cpu_tensor->data(), tensor->data(), tensor->nbytes());
        synchronize_stream_func(stream);

        print_tensor_elem(cpu_tensor);
        diopiDestoryTensor(ctx, cpu_tensor);
    }

    if (nbytes > 0) {
        printf(",\n\tshape:[");
        int64_t i = 0;
        for (i = 0; i < shape.len; i++) {
            printf("%d", (int)shape.data[i]);
            if (i < shape.len - 1) {
                printf(",");
            }
        }
        printf("]");

        printf(", stride:[");
        for (i = 0; i < stride.len; i++) {
            printf("%lld", stride.data[i]);
            if (i < stride.len - 1) {
                printf(",");
            }
        }
        printf("]");
        printf(", nbytes=%lld", nbytes);
        printf(", numel=%lld", numel);
        printf(", dtype=%s", diopi_dtype_to_str(dtype));
        printf(", device=%s", device_to_str(device));
    }

    printf(")\n");
    DIOPIRT_LOG_LEVEL = DIOPIRT_LOG_LEVEL_TEMP;
    return diopiSuccess;
}


}
