/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/

#ifndef _DIOPI_REFERENCE_IMPLCUDA_HELPER_HPP_
#define _DIOPI_REFERENCE_IMPLCUDA_HELPER_HPP_

#include <diopi/diopirt.h>
#include <cuda_runtime.h>


#define DIOPI_CALL(Expr){                                                               \
    if( diopiSuccess != Expr ){                                                         \
        return Expr;                                                                    \
    }}

extern "C" void set_error_string(const char *err);

namespace impl {

namespace cuda {

template<typename T>
struct DataType;

template<>
struct DataType<diopiTensorHandle_t> {
    using type = void*;

    static void* data(diopiTensorHandle_t& tensor) {
        void *data;
        diopiGetTensorData(&tensor, &data);
        return data;
    }
};

template<>
struct DataType<const diopiTensorHandle_t> {
    using type = const void*;

    static const void* data(const diopiTensorHandle_t& tensor) {
        const void *data;
        diopiGetTensorDataConst(&tensor, &data);
        return data;
    }
};

template<typename T>
class DiopiTensor final {
public:
    explicit DiopiTensor(T& tensor) : tensor_(tensor) {}

    diopiDevice_t device() const {
        diopiDevice_t device;
        diopiGetTensorDevice(tensor_, &device);
        return device;
    }
    diopiDtype_t dtype() const {
        diopiDtype_t dtype;
        diopiGetTensorDtype(tensor_, &dtype);
        return dtype;
    }

    const diopiSize_t& shape(){
        diopiGetTensorShape(tensor_, &shape_);
        return shape_;
    }
    const diopiSize_t& stride(){
        diopiGetTensorStride(tensor_, &stride_);
        return stride_;
    }

    int64_t numel() const {
        int64_t numel;
        diopiGetTensorNumel(tensor_, &numel);
        return numel;
    }
    int64_t elemsize() const {
        int64_t elemsize;
        diopiGetTensorElemSize(tensor_, &elemsize);
        return elemsize;
    }

    typename DataType<T>::type data() {
        return DataType<T>::data(tensor_);
    }

protected:
    T& tensor_;

    diopiSize_t shape_;
    diopiSize_t stride_;
};

template<typename T>
auto makeTensor(T& tensor) -> DiopiTensor<T> {
    return DiopiTensor<T>(tensor);
}

inline cudaStream_t getStream(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    return static_cast<cudaStream_t>(stream_handle);
}

}  // namespace cuda

}  // namespace impl

#endif  // _DIOPI_REFERENCE_IMPLCUDA_HELPER_HPP_
