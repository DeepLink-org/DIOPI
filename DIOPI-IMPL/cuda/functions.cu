#include <diopi/functions.h>
#include <cuda_runtime.h>


#define dispatch_dtype(fun, dtype, gridSize, blockSize, ...)                                     \
    if (diopi_dtype_int32 == dtype) {                                                            \
        fun<int32_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                      \
    }                                                                                            \
    else if (diopi_dtype_uint32 == dtype) {                                                      \
        fun<uint32_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                     \
    }                                                                                            \
    else if (diopi_dtype_int16 == dtype) {                                                       \
        fun<int16_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                      \
    }                                                                                            \
    else if (diopi_dtype_uint16 == dtype) {                                                      \
        fun<uint16_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                     \
    }                                                                                            \
    else if (diopi_dtype_int8 == dtype) {                                                        \
        fun<int8_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                       \
    }                                                                                            \
    else if (diopi_dtype_uint8 == dtype) {                                                       \
        fun<uint8_t><<<gridSize, blockSize>>>(__VA_ARGS__);                                      \
    }                                                                                            \
    else if (diopi_dtype_float32 == dtype) {                                                     \
        fun<float><<<gridSize, blockSize>>>(__VA_ARGS__);                                        \
    }                                                                                            \
    else if (diopi_dtype_float64 == dtype) {                                                     \
        fun<double><<<gridSize, blockSize>>>(__VA_ARGS__);                                       \
    }                                                                                            \
    else if (diopi_dtype_bool == dtype) {                                                        \
        fun<bool><<<gridSize, blockSize>>>(__VA_ARGS__);                                         \
    }                                                                                            \
    else {                                                                                       \
        fprintf(stderr, "%s:%s: %s<%s %d><<<%d,%d>>>(%s)", __FILE__, __FUNCTION__, #fun, #dtype, \
                dtype, gridSize, blockSize, #__VA_ARGS__);                                       \
        return diopiDtypeNotSupported;                                                           \
    }


template<typename T> __global__
void vecAdd(const void* a, const void* b, void* c, const int numel)
{
    int      id = blockIdx.x * blockDim.x + threadIdx.x;
    const T* A  = static_cast<const T*>(a);
    const T* B  = static_cast<const T*>(b);
    T*       C  = static_cast<T*>(c);
    if (id < numel) {
        C[id] = A[id] + B[id];
    }
}

extern "C" diopiError_t add(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other)
{
    diopiDevice_t       device;
    diopiDtype_t        dtype;
    int64_t             numel;
    diopiSize_t         shape;
    diopiSize_t         stride;
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    diopiGetTensorDevice(input, &device);
    diopiGetTensorDtype(input, &dtype);
    diopiGetTensorNumel(input, &numel);

    diopiGetTensorShape(input, &shape);
    diopiGetTensorStride(input, &stride);

    const void* input_tensor_data_ptr;
    diopiGetTensorDataConst(input, &input_tensor_data_ptr);
    const void* other_tensor_data_ptr;
    diopiGetTensorDataConst(other, &other_tensor_data_ptr);

    void* out_tensor_data_ptr;
    diopiGetTensorData(out, &out_tensor_data_ptr);

    int blockSize = 256;
    int gridSize  = (numel + blockSize - 1) / blockSize;
    dispatch_dtype(vecAdd,
                dtype,
                gridSize,
                blockSize,
                input_tensor_data_ptr,
                other_tensor_data_ptr,
                out_tensor_data_ptr,
                numel);
    return diopiSuccess;
}


template<typename T> __global__
void vecFill(void* a, const T value, const int numel)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    T*  A  = static_cast<T*>(a);
    if (id < numel) {
        A[id] = value;
    }
}

extern "C" diopiError_t fill(diopiContextHandle_t ctx, diopiTensorHandle_t tensor, const float value)
{
    diopiDevice_t device;
    diopiDtype_t  dtype;
    int64_t       numel;

    diopiGetTensorDevice(tensor, &device);
    diopiGetTensorDtype(tensor, &dtype);
    diopiGetTensorNumel(tensor, &numel);

    void* tensor_data_ptr;
    diopiGetTensorData(tensor, &tensor_data_ptr);

    if (diopi_host == device) {
        return diopiErrorOccurred;
    } else {
        int blockSize = 256;
        int gridSize  = (numel + blockSize - 1) / blockSize;
        dispatch_dtype(vecFill, dtype, gridSize, blockSize, tensor_data_ptr, value, numel);
    }

    return diopiSuccess;
}
