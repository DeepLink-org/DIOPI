#ifndef _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_
#define _DIOPI_REFERENCE_IMPLTORCH_ATEN_HPP_

#include <vector>

#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include <diopi/diopirt.h>

using diopi_tensor_list = std::vector<diopiTensorHandle_t>;

namespace impl {

namespace aten {

caffe2::TypeMeta getATenType(diopiDtype_t dt) {
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

c10::DeviceType getATenDevice(diopiDevice_t device) {
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
    if (tensor == nullptr) return at::Tensor();

    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    caffe2::TypeMeta atType = getATenType(dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    c10::DeviceType atDevice = getATenDevice(device);

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

at::IntArrayRef buildAtIntArray(const diopiSize_t* size) {
    return at::IntArrayRef(size->data, size->len);
}

at::IntArrayRef buildAtIntArray(diopiSize_t size) {
    return at::IntArrayRef(size.data, size.len);
}

decltype(auto) buildAtTensorList(const diopiTensorHandle_t* tensors, int64_t numTensors) {
    std::vector<at::Tensor> vecAtTensor;
    for (size_t i = 0; i < numTensors; ++i) {
        vecAtTensor.emplace_back(buildAtTensor(tensors[i]));
    }
    return vecAtTensor;
}

void updateATen2Tensor(diopiContextHandle_t ctx, const at::Tensor& atOut, diopiTensorHandle_t out) {
    // TODO(fengsibo): add device and nbytes check
    void* src = atOut.data_ptr();
    size_t nbytes = atOut.nbytes();
    void* dst = nullptr;
    diopiGetTensorData(&out, &dst);
    cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice);
}

template<typename TupleT, std::size_t N>
struct UpdateTupleATen {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts,
            diopi_tensor_list& outs) {
        UpdateTupleATen<TupleT, N - 1>::update(ctx, atOuts, outs);
        updateATen2Tensor(ctx, std::get<N - 1>(atOuts), outs.at(N - 1));
    }
};

template<typename TupleT>
struct UpdateTupleATen<TupleT, 1> {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts,
            std::vector<diopiTensorHandle_t>& outs) {
        updateATen2Tensor(ctx, std::get<0>(atOuts), outs.at(0));
    }
};

template<typename TupleT>
void updateATen2Tensor(diopiContextHandle_t ctx, TupleT& atOuts, diopi_tensor_list& outs) {
    constexpr size_t tupleSize = std::tuple_size<TupleT>::value;
    UpdateTupleATen<TupleT, tupleSize>::update(ctx, atOuts, outs);
}

void updateATen2Tensor(diopiContextHandle_t ctx, std::vector<at::Tensor>& atOuts, diopi_tensor_list& outs) {
    for (size_t i = 0; i < atOuts.size(); ++i) {
        updateATen2Tensor(ctx, atOuts.at(i), outs.at(i));
    }
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopiTensorHandle_t out, Args&&... args) {
    at::Tensor atOut = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOut, out);
}

template<typename Func, typename ...Args>
void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopi_tensor_list& outs, Args&&... args) {
    auto atOuts = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOuts, outs);
}

template<typename Func, typename ...Args>
void invokeATenFuncInp(diopiContextHandle_t ctx, Func func, Args&&... args) {
    func(std::forward<Args>(args)...);
}

}  // namespace aten

}  // namespace impl

#endif
