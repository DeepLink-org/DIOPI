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

#include "torch_npu/csrc/framework/DIOPIAdapter.h"


#define LOG_LINE_INFO() std::cerr << __FILE__ << ":" << __LINE__ << ": ";

#define GET_ARGS_NUM

#define BUILD_ATEN_ARG0() ;

#define BUILD_ATEN_ARG1(x) \
    auto at_##x = impl::aten::buildATen(x);

#define BUILD_ATEN_ARG2(x, y) \
    auto at_##x = impl::aten::buildATen(x); \
    auto at_##y = impl::aten::buildATen(y);

#define BUILD_ATEN_ARG3(x, y, z) \
    auto at_##x = impl::aten::buildATen(x); \
    auto at_##y = impl::aten::buildATen(y); \
    auto at_##z = impl::aten::buildATen(z);

#define BUILD_ATEN_ARG4(x1, x2, x3, x4) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4);

#define BUILD_ATEN_ARG5(x1, x2, x3, x4, x5) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5);

#define BUILD_ATEN_ARG6(x1, x2, x3, x4, x5, x6) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5); \
    auto at_##x6 = impl::aten::buildATen(x6);

#define BUILD_ATEN_ARG7(x1, x2, x3, x4, x5, x6, x7) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5); \
    auto at_##x6 = impl::aten::buildATen(x6); \
    auto at_##x7 = impl::aten::buildATen(x7);

#define BUILD_ATEN_ARG8(x1, x2, x3, x4, x5, x6, x7, x8) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5); \
    auto at_##x6 = impl::aten::buildATen(x6); \
    auto at_##x7 = impl::aten::buildATen(x7); \
    auto at_##x8 = impl::aten::buildATen(x8);

#define BUILD_ATEN_ARG9(x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5); \
    auto at_##x6 = impl::aten::buildATen(x6); \
    auto at_##x7 = impl::aten::buildATen(x7); \
    auto at_##x8 = impl::aten::buildATen(x8); \
    auto at_##x9 = impl::aten::buildATen(x9);

#define BUILD_ATEN_ARG10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
    auto at_##x1 = impl::aten::buildATen(x1); \
    auto at_##x2 = impl::aten::buildATen(x2); \
    auto at_##x3 = impl::aten::buildATen(x3); \
    auto at_##x4 = impl::aten::buildATen(x4); \
    auto at_##x5 = impl::aten::buildATen(x5); \
    auto at_##x6 = impl::aten::buildATen(x6); \
    auto at_##x7 = impl::aten::buildATen(x7); \
    auto at_##x8 = impl::aten::buildATen(x8); \
    auto at_##x9 = impl::aten::buildATen(x9); \
    auto at_##x10 = impl::aten::buildATen(x10);


#define PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,N,...) N
#define PRIVATE_MACRO_VAR_ARGS_IMPL(args) PRIVATE_MACRO_VAR_ARGS_IMPL_COUNT args

#define COUNT_MACRO_VARR(...) PRIVATE_MACRO_VAR_ARGS_IMPL((__VA_ARGS__,14,13,12,11,10,9,8,7,6,5,4,3,2,1))

#define PRIVATE_CONCAT_STR2(x, y) x##y
#define PRIVATE_CONCAT_STR1(x, y) PRIVATE_CONCAT_STR2(x, y)
#define PRIVATE_CONCAT_STR(x, y) PRIVATE_CONCAT_STR1(x, y)

#define BUILD_ATEN_ARGS(...) \
    PRIVATE_CONCAT_STR(BUILD_ATEN_ARG, COUNT_MACRO_VARR(__VA_ARGS__))(__VA_ARGS__)



#define BEGIN_CALL_ACL_OP(...)                                          \
    std::cout<<__FILE__<<":"<<__LINE__<<" :"<<__FUNCTION__<<std::endl;  \
    impl::aten::setCurCtx(ctx);                                         \
    BUILD_ATEN_ARGS(__VA_ARGS__)

#define END_CALL_ACL_OP()                                             \
    impl::aten::unsetCurCtx();                                        \
    return diopiSuccess;

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

inline at::Scalar buildATen(const diopiScalar_t* scalar) {
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

inline at::IntArrayRef buildATen(const diopiSize_t* size) { return at::IntArrayRef(size->data, size->len); }

inline at::IntArrayRef buildATen(diopiSize_t size) { return at::IntArrayRef(size.data, size.len); }

inline c10::OptionalIntArrayRef buildATen(diopiSize_t* size_ptr) {
    if (size_ptr) {
        return buildATen(*size_ptr);
    }
    return c10::nullopt;
}


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
