#ifndef _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_
#define _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_

#include <vector>

#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include <diopi/diopirt.h>


caffe2::TypeMeta getATensorType(diopiDtype_t dt) {
    switch (dt) {
    case diopi_dtype_bool:
        return caffe2::TypeMeta::Make<bool>();
    case diopi_dtype_uint8:
        return caffe2::TypeMeta::Make<uint8_t>();
    case diopi_dtype_int8:
        return caffe2::TypeMeta::Make<int8_t>();
    case diopi_dtype_int16:
        return caffe2::TypeMeta::Make<int16_t>();
    case diopi_dtype_uint16:
        return caffe2::TypeMeta::Make<uint16_t>();
    case diopi_dtype_int32:
    case  diopi_dtype_uint32:
        return caffe2::TypeMeta::Make<int32_t>();
    case diopi_dtype_int64:
    case diopi_dtype_uint64:
        return caffe2::TypeMeta::Make<int64_t>();
        return caffe2::TypeMeta::Make<uint64_t>();
    case diopi_dtype_float32:
        return caffe2::TypeMeta::Make<float>();
    case diopi_dtype_float64:
        return caffe2::TypeMeta::Make<double>();
    case diopi_dtype_float16:
        return caffe2::TypeMeta::Make<at::Half>();
    case diopi_dtype_bfloat16:
        return caffe2::TypeMeta::Make<at::BFloat16>();
    default:
        std::fprintf(stderr, "Dtype not supported");
    }
}

c10::DeviceType getATensorDevice(diopiDevice_t device) {
    if (device == diopi_host) {
        return c10::DeviceType::CPU;
    } else if (device == diopi_device) {
        return c10::DeviceType::CUDA;
    } else {
        std::fprintf(stderr, "Device not supported");
    }
}

at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes,
        at::IntArrayRef strides, const std::function<void(void*)>& deleter,
        at::Allocator* allocator, const at::TensorOptions& options) {
    auto device =
        at::globalContext().getDeviceFromPtr(data, options.device().type());
    if (options.device().has_index()) {
        assert(options.device() == device);
    }

    auto storage = at::Storage(
        at::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
        c10::InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
        allocator, false);
    return at::empty({0}, options).set_(storage, 0, sizes, strides);
}

at::Tensor buildAtTensor(diopiTensorHandle_t tensor) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    caffe2::TypeMeta atType = getATensorType(dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    c10::DeviceType atDevice = getATensorDevice(device);

    void* data = nullptr;
    diopiGetTensorData(&tensor, &data);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    at::IntArrayRef atDims(shape.data, shape.len);

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    at::IntArrayRef atStrides(stride.data, stride.len);

    auto options = at::TensorOptions(atDevice).dtype(atType);
    int64_t numel = 0;
    diopiGetTensorNumel(tensor, &numel);
    if (0 == numel) {
        return at::empty(atDims, options);
    } else {
        at::Allocator* allocator = nullptr;
        return fromPreAllocated(data, atDims,
            atStrides, [](void*){}, allocator, options);
    }
}

at::Scalar buildAtScalar(const diopiTensorHandle_t input, const diopiScalar_t* scalar) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (dtype <= 7) {
        return scalar->ival;
    } else if (dtype <= 11) {
        return scalar->fval;
    } else {
        std::fprintf(stderr, "Dtype not supported");
    }
}

void updateATen2Tensor(diopiContextHandle_t ctx, const at::Tensor& atOut, diopiTensorHandle_t out) {
    // TODO(fengsibo): add device and nbytes check
    void* src = atOut.data_ptr();
    size_t nbytes = atOut.nbytes();
    void* dst = nullptr;
    diopiGetTensorData(&out, &dst);
    cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice);
}

template<size_t N, typename TupleT>
struct TupleToList {
    static void copy(diopiContextHandle_t ctx, const TupleT& aouts,
            std::vector<diopiTensorHandle_t>& outs, size_t idx) {
        auto& aout = std::get<N>(aouts);
        if (aout.defined()) {
            updateATen2Tensor(ctx, aout, outs[idx]);
            --idx;
        }
        TupleToList<N - 1, TupleT>::copy(ctx, aouts, outs, idx);
    }

    static void count(const TupleT& aouts, size_t& cnt) {
        auto& t = std::get<N>(aouts);
        if (t.defined()) ++cnt;
        TupleToList<N - 1, TupleT>::count(aouts, cnt);
    }
};

template<typename TupleT>
void updateATen2Tensor(diopiContextHandle_t ctx, TupleT& atOuts, std::vector<diopiTensorHandle_t>& out) {
    constexpr size_t tupleSize = std::tuple_size<TupleT>::value;
    size_t count = 0;
    std::cout << tupleSize << std::endl;
    // TupleToList<tupleSize - 1, TupleT>::count(atOuts, count);
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopiTensorHandle_t out, Args&&... args) {
    at::Tensor atOut = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOut, out);
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, std::vector<diopiTensorHandle_t>& outs, Args&&... args) {
    auto atOuts = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOuts, outs);
}

template<typename Func, typename ...Args>
void invokeATenFuncInp(diopiContextHandle_t ctx, Func func, Args&&... args) {
    at::Tensor atOut = func(std::forward<Args>(args)...);
}

#endif
