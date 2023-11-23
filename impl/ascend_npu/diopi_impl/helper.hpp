/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_HELPER_HPP_
#define IMPL_TORCH_HELPER_HPP_
#include <ATen/ATen.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>
#include "error.hpp"


#define LOG_LINE_INFO() std::cerr << __FILE__ << ":" << __LINE__ << ": ";


#define BEGIN_CALL_ACL_OP()   \
    std::cout<<__FILE__<<":"<<__LINE__<<" :"<<__FUNCTION__<<std::endl;

inline void logError() { std::cerr << std::endl; }

template <typename First, typename... Rest>
void logError(First&& first, Rest&&... rest) {
    std::cerr << std::forward<First>(first);
    logError(std::forward<Rest>(rest)...);
}

template <typename... Types>
void set_last_error_string(const char* szFmt, Types&&... args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

#define ATEN_NOT_IMPLEMENT()                                                                                         \
    LOG_LINE_INFO()                                                                                                  \
    logError("NotImplementError: function ", __FUNCTION__, " is not implemented for torch version ", TORCH_VERSION); \
    set_last_error_string("NotImplementError: function %s is not implemented for torch version %d" __FUNCTION__, TORCH_VERSION);

#define NOT_SUPPORTED(str) set_last_error_string("NotSupported: %s at %s:%d", str, __FILE__, __LINE__);

#define DIOPI_CHECK(cond, str)                                         \
    if (!(cond)) {                                                     \
        set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
        return diopiErrorOccurred;                                     \
    }

#define DIOPI_CHECK_PTR(ptr)                                                                     \
    if (ptr == nullptr) {                                                                        \
        set_last_error_string("NotSupported: %s is nullptr at %s:%d", #ptr, __FILE__, __LINE__); \
        return diopiErrorOccurred;                                                               \
    }

using diopi_tensor_list = std::vector<diopiTensorHandle_t>;
extern thread_local diopiContextHandle_t context;

#if 0
namespace c10 {

namespace cuda {

// Note: this is a overloaded aten function to get the stream from context.
// For original functions, please refer to https://github.com/pytorch/pytorch/blob/v1.10.0/c10/cuda/CUDAStream.cpp.
inline CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
    if (device_index == -1) {
        device_index = current_device();
    }
    if (context) {
        diopiStreamHandle_t stream_handle;
        diopiGetStream(context, &stream_handle);
        return getStreamFromExternal(static_cast<cudaStream_t>(stream_handle), device_index);
    } else {
        return getDefaultCUDAStream(device_index);
    }
}

}  // namespace cuda
}  // namespace c10
#endif

namespace impl {

namespace aten {

inline void setCurCtx(diopiContextHandle_t ctx) {
    context = ctx;
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    //c10::cuda::CUDAStream cur_stream = c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream_handle), c10::cuda::current_device());
    //c10::cuda::setCurrentCUDAStream(cur_stream);
    // Here, we set the current stream of cuda to the stream of diopi, but when the context is unloaded, it is not restored.
    // The main reason is that the current stream of cuda is not used. However, there may be hidden bugs, which will be optimized later.
}

inline void unsetCurCtx() { context = nullptr; }

inline void sync(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    //cudaStreamSynchronize(static_cast<cudaStream_t>(stream_handle));
}

inline caffe2::TypeMeta getATenType(diopiDtype_t dt) {
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
            return caffe2::TypeMeta::Make<int32_t>();
        // case  diopi_dtype_uint32: // can not find symbol for uint32_t
        //     return caffe2::TypeMeta::Make<uint32_t>();
        case diopi_dtype_int64:
            return caffe2::TypeMeta::Make<int64_t>();
        // case diopi_dtype_uint64:  // can not find symbol for uint64_t
        //     return caffe2::TypeMeta::Make<uint64_t>();
        case diopi_dtype_float32:
            return caffe2::TypeMeta::Make<float>();
        case diopi_dtype_float64:
            return caffe2::TypeMeta::Make<double>();
        case diopi_dtype_float16:
            return caffe2::TypeMeta::Make<at::Half>();
        case diopi_dtype_bfloat16:
            return caffe2::TypeMeta::Make<at::BFloat16>();
        case diopi_dtype_complex64:
            return caffe2::TypeMeta::Make<c10::complex<float>>();
        case diopi_dtype_complex128:
            return caffe2::TypeMeta::Make<c10::complex<double>>();
        default:
            NOT_SUPPORTED("diopi dytpe");
            return caffe2::TypeMeta();
    }
}

inline diopiDtype_t getDIOPITensorType(at::Tensor& input) {
    switch (input.scalar_type()) {
        case at::ScalarType::Bool:
            return diopi_dtype_bool;
        case at::ScalarType::Char:
            return diopi_dtype_int8;
        case at::ScalarType::Byte:
            return diopi_dtype_uint8;
        case at::ScalarType::Short:
            return diopi_dtype_int16;
        case at::ScalarType::Int:
            return diopi_dtype_int32;
        case at::ScalarType::Long:
            return diopi_dtype_int64;
        case at::ScalarType::Half:
            return diopi_dtype_float16;
        case at::ScalarType::BFloat16:
            return diopi_dtype_bfloat16;
        case at::ScalarType::Float:
            return diopi_dtype_float32;
        case at::ScalarType::Double:
            return diopi_dtype_float64;
        default:
            NOT_SUPPORTED("aten dtype");
            return diopi_dtype_unsupported;
    }
}

inline diopiDevice_t getDIOPIDevice(c10::DeviceType device) {
    if (device == c10::DeviceType::CPU) {
        return diopi_host;
    }
    return diopi_device;
}

inline c10::DeviceType getATenDevice(diopiDevice_t device) {
    if (device == diopi_host) {
        return c10::DeviceType::CPU;
    }
    return c10::DeviceType::XPU;
}

inline at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, const std::function<void(void*)>& deleter,
                                   at::Allocator* allocator, const at::TensorOptions& options) {
    //auto device = at::globalContext().getDeviceFromPtr(data, options.device().type());
    auto device = options.device();
    if (options.device().has_index()) {
        assert(options.device() == device);
    }

    auto storage = at::Storage(at::Storage::use_byte_size_t(),
                               at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
                               c10::InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
                               allocator,
                               false);

    at::TensorOptions new_options = options.device(device);
    return at::empty({0}, new_options).set_(storage, 0, sizes, strides);
}

inline at::Tensor buildATen(diopiTensorHandle_t tensor) {
    if (tensor == nullptr) return at::Tensor();
    return *reinterpret_cast<at::Tensor*>(tensor);
}

inline const at::Tensor buildATen(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) return at::Tensor();
    return *reinterpret_cast<const at::Tensor*>(tensor);
}

inline bool isInt(const diopiScalar_t* scalar) { return scalar->stype <= 7; }

inline bool isFloat(const diopiScalar_t* scalar) { return scalar->stype > 7; }

inline at::Scalar buildAtScalar(const diopiScalar_t* scalar) {
    if (scalar == nullptr) {
        NOT_SUPPORTED("scalar is null ptr, we use temporarily zero");
        return at::Scalar();
    }
    if (isInt(scalar)) {
        int64_t ival = scalar->ival;
        return ival;
    } else {
        double fval = scalar->fval;
        return fval;
    }
}

inline at::IntArrayRef buildAtIntArray(const diopiSize_t* size) { return at::IntArrayRef(size->data, size->len); }

inline at::IntArrayRef buildAtIntArray(diopiSize_t size) { return at::IntArrayRef(size.data, size.len); }

template <typename T>
inline decltype(auto) buildATenList(T* tensors, int64_t numTensors) {
    std::vector<at::Tensor> vecAtTensor;
    for (size_t i = 0; i < numTensors; ++i) {
        vecAtTensor.emplace_back(buildATen(tensors[i]));
    }
    return vecAtTensor;
}

inline void updateATen2Tensor(diopiContextHandle_t ctx, const at::Tensor& atOut, diopiTensorHandle_t out) {
    // TODO(fengsibo): add device and nbytes check
    if (out != nullptr) {
        at::Tensor atOutput = buildATen(out);
        atOutput.reshape_as(atOut).copy_(atOut, true);
    }
}

template <typename TupleT, std::size_t N>
struct UpdateTupleATen {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts, diopi_tensor_list& outs) {
        UpdateTupleATen<TupleT, N - 1>::update(ctx, atOuts, outs);
        updateATen2Tensor(ctx, std::get<N - 1>(atOuts), outs.at(N - 1));
    }
};

template <typename TupleT>
struct UpdateTupleATen<TupleT, 1> {
    static void update(diopiContextHandle_t ctx, TupleT& atOuts, std::vector<diopiTensorHandle_t>& outs) {
        updateATen2Tensor(ctx, std::get<0>(atOuts), outs.at(0));
    }
};

template <typename TupleT>
inline void updateATen2Tensor(diopiContextHandle_t ctx, TupleT& atOuts, diopi_tensor_list& outs) {
    constexpr size_t tupleSize = std::tuple_size<TupleT>::value;
    UpdateTupleATen<TupleT, tupleSize>::update(ctx, atOuts, outs);
}

inline void updateATen2Tensor(diopiContextHandle_t ctx, std::vector<at::Tensor>& atOuts, diopi_tensor_list& outs) {
    for (size_t i = 0; i < atOuts.size(); ++i) {
        updateATen2Tensor(ctx, atOuts.at(i), outs.at(i));
    }
}

template <typename Func, typename... Args>
inline void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopiTensorHandle_t out, Args&&... args) {
    at::Tensor atOut = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOut, out);
}

template <typename Func, typename... Args>
inline void invokeATenFuncRet(diopiContextHandle_t ctx, Func func, diopi_tensor_list& outs, Args&&... args) {
    auto atOuts = func(std::forward<Args>(args)...);
    updateATen2Tensor(ctx, atOuts, outs);
}

template <typename Func, typename... Args>
inline void invokeATenFuncInp(diopiContextHandle_t ctx, Func func, Args&&... args) {
    func(std::forward<Args>(args)...);
}

inline void buildDiopiTensor(diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t* out) {
    at::IntArrayRef atSize = input.sizes();
    at::IntArrayRef atStride = input.strides();
    diopiSize_t size{atSize.data(), static_cast<int64_t>(atSize.size())};
    diopiSize_t stride{atStride.data(), static_cast<int64_t>(atStride.size())};
    diopiDtype_t dtype = getDIOPITensorType(input);
    diopiDevice_t device = getDIOPIDevice(input.device().type());
    diopiRequireTensor(ctx, out, &size, &stride, dtype, device);
    updateATen2Tensor(ctx, input, *out);
}

// new cuda generator and pass dipu generator state into cuda generator state
inline at::Generator buildGenerator(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t generator) {
    at::Generator gen;
    diopiTensorHandle_t state_handle = nullptr;
    diopiGeneratorGetState(ctx, generator, &state_handle);
    auto state = impl::aten::buildATen(state_handle);
    {
        std::lock_guard<std::mutex> lock(gen.mutex());
        gen.set_state(state);
    }
    return gen;
}

inline void updateGeneratorHandleState(diopiContextHandle_t ctx, at::Generator& cuda_gen, diopiGeneratorHandle_t generator) {
    at::Tensor new_state;
    {
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        new_state = cuda_gen.get_state();
    }
    diopiTensorHandle_t new_state_handle = nullptr;
    buildDiopiTensor(ctx, new_state, &new_state_handle);
    diopiGeneratorSetState(generator, new_state_handle);
}


}  // namespace aten

}  // namespace impl

#endif  // IMPL_TORCH_HELPER_HPP_
