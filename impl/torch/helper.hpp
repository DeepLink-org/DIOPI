/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_TORCH_HELPER_HPP_
#define IMPL_TORCH_HELPER_HPP_
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "error.hpp"
#include "impl_functions.hpp"

#define TORCH_MM_VERSION (TORCH_VERSION_MAJOR * 1000 + TORCH_VERSION_MINOR * 10)
#define TORCH_1_7_MM_VERSION 1070
#define TORCH_1_8_MM_VERSION 1080
#define TORCH_1_9_MM_VERSION 1090
#define TORCH_1_10_MM_VERSION 1100
#define TORCH_1_11_MM_VERSION 1110
#define TORCH_1_12_MM_VERSION 1120

#define LOG_LINE_INFO() std::cerr << __FILE__ << ":" << __LINE__ << ": ";

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

namespace impl {

namespace aten {

inline void setCurCtx(diopiContextHandle_t ctx) {
    context = ctx;
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    c10::cuda::CUDAStream cur_stream = c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream_handle), c10::cuda::current_device());
    c10::cuda::setCurrentCUDAStream(cur_stream);
    // Here, we set the current stream of cuda to the stream of diopi, but when the context is unloaded, it is not restored.
    // The main reason is that the current stream of cuda is not used. However, there may be hidden bugs, which will be optimized later.
}

inline void unsetCurCtx() { context = nullptr; }

inline void sync(diopiContextHandle_t ctx) {
    diopiStreamHandle_t stream_handle;
    diopiGetStream(ctx, &stream_handle);
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_handle));
}

caffe2::TypeMeta getATenType(diopiDtype_t dt);

diopiDtype_t getDIOPITensorType(at::Tensor& input);

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
    return c10::DeviceType::CUDA;
}

at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, const std::function<void(void*)>& deleter, at::Allocator* allocator,
                            const at::TensorOptions& options);

template <typename T>
at::Tensor buildATen(T tensor);

inline bool isInt(const diopiScalar_t* scalar) { return scalar->stype <= 7; }

inline bool isFloat(const diopiScalar_t* scalar) { return scalar->stype > 7; }

at::Scalar buildAtScalar(const diopiScalar_t* scalar);

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

void buildDiopiTensor(diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t* out);

// new cuda generator and pass dipu generator state into cuda generator state
at::Generator buildGenerator(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t generator);

void updateGeneratorHandleState(diopiContextHandle_t ctx, at::Generator& cuda_gen, diopiGeneratorHandle_t generator);

inline c10::optional<c10::string_view> getRoundingMode(diopiRoundMode_t rounding_mode) {
    switch (rounding_mode) {
        case (RoundModeNone):
            return c10::nullopt;
        case (RoundModeTrunc):
            return "trunc";
        case (RoundModeFloor):
            return "floor";
        case (RoundModeEND):
            return "";
        default:
            NOT_SUPPORTED("diopi round mode");
            return "";
    }
}

at::Tensor nllLossNdBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight, int64_t reduction,
                             int64_t ignore_index);

at::Tensor crossEntropyLossProbTargetBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight,
                                              int64_t reduction, double label_smoothing);

at::Tensor crossEntropyLossLabelSmoothingBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight,
                                                  int64_t reduction, int64_t ignore_index, double label_smoothing);

}  // namespace aten

}  // namespace impl

#endif  // IMPL_TORCH_HELPER_HPP_
