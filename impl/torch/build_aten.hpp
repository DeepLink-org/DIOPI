#pragma once

#include <ATen/Context.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "diopi/diopirt.h"

namespace impl::aten {

// This class is a wrapper around an at::Tensor, which changes the device and dispatch key of the binded at::TensorImpl from any device (e.g. XPU) to CUDA, and
// revert it back when the wrapper is destroyed.
// The wrapper is designed to be implicitly converted to an at::Tensor (object slicing), so that it can be used in place of an at::Tensor.
class UnsafelyDeviceChangedTensorWrapper : public at::Tensor {
public:
    static UnsafelyDeviceChangedTensorWrapper createFromTensor(const at::Tensor& tensor) { return UnsafelyDeviceChangedTensorWrapper(tensor); }
    UnsafelyDeviceChangedTensorWrapper() = default;
    ~UnsafelyDeviceChangedTensorWrapper();
    UnsafelyDeviceChangedTensorWrapper(const UnsafelyDeviceChangedTensorWrapper& other) : at::Tensor(other) {}
    UnsafelyDeviceChangedTensorWrapper(UnsafelyDeviceChangedTensorWrapper&& other) : at::Tensor(std::move(other)) { saveForRevert_.swap(other.saveForRevert_); }
    UnsafelyDeviceChangedTensorWrapper& operator=(const UnsafelyDeviceChangedTensorWrapper& other) = delete;
    UnsafelyDeviceChangedTensorWrapper& operator=(UnsafelyDeviceChangedTensorWrapper&& other) {
        at::Tensor::operator=(std::move(other));
        saveForRevert_.swap(other.saveForRevert_);
        return *this;
    }
    UnsafelyDeviceChangedTensorWrapper& operator=(const at::Tensor& other) {
        at::Tensor::operator=(other);
        return *this;
    }
    UnsafelyDeviceChangedTensorWrapper& operator=(at::Tensor&& other) {
        at::Tensor::operator=(std::move(other));
        return *this;
    }

private:
    explicit UnsafelyDeviceChangedTensorWrapper(const at::Tensor& tensor);
    using TensorImplAndDevice = std::pair<at::TensorImpl*, at::Device>;
    static void setTensorImplDeviceUnsafe(const TensorImplAndDevice& tensorAndDevice);
    c10::optional<TensorImplAndDevice> saveForRevert_ = c10::nullopt;
};

// WARNING: This function is UNSAFE. It is the caller's responsibility to ensure that:
//   1. The returned wrapper is not destroyed when its sliced at::Tensor is still in use in DIOPI.
//   2. The input diopiConstTensorHandle_t is actually a reinterpret_cast of an at::Tensor*.
//   3. The input tensor is only used in one thread.
[[nodiscard]] UnsafelyDeviceChangedTensorWrapper buildATenUnsafe(diopiConstTensorHandle_t tensor);

[[nodiscard]] at::Tensor buildATenSafe(diopiConstTensorHandle_t tensor);

[[nodiscard]] inline auto buildATen(diopiConstTensorHandle_t tensor) {
#if 1
    return buildATenUnsafe(tensor);
#else
    return buildATenSafe(tensor);
#endif
}

template <typename T>
[[nodiscard]] auto buildATenList(T* tensors, int64_t numTensors) {
    using TensorType = decltype(buildATen(std::declval<diopiConstTensorHandle_t>()));
    c10::SmallVector<TensorType, 4> vecAtTensor;
    vecAtTensor.reserve(numTensors);
    std::transform(tensors, tensors + numTensors, std::back_inserter(vecAtTensor), [](auto tensor) { return buildATen(tensor); });
    return vecAtTensor;
}

// These macros is designed to avoid early destruction of the wrapper when build optional at::Tensor.
#define DIOPI_IMPL_BUILD_ATEN_LIST(atTensor, diopiTensors, numTensors)                                                                  \
    auto atTensor##__MAYBE_WRAPPER = ::impl::aten::buildATenList(diopiTensors, numTensors);                                             \
    c10::SmallVector<at::Tensor, 4> atTensor;                                                                                           \
    atTensor.reserve(numTensors);                                                                                                       \
    std::transform(atTensor##__MAYBE_WRAPPER.begin(), atTensor##__MAYBE_WRAPPER.end(), std::back_inserter(atTensor), [](auto& tensor) { \
        return static_cast<at::Tensor>(tensor);                                                                                         \
    });
#define DIOPI_IMPL_BUILD_ATEN_OPTIONAL(atTensor, diopiTensor)              \
    auto atTensor##__MAYBE_WRAPPER = ::impl::aten::buildATen(diopiTensor); \
    c10::optional<at::Tensor> atTensor;                                    \
    if (atTensor##__MAYBE_WRAPPER.defined()) {                             \
        atTensor = atTensor##__MAYBE_WRAPPER;                              \
    }
#define DIOPI_IMPL_BUILD_ATEN_OPTIONAL_LIST(atTensor, diopiTensors, numTensors)                                                         \
    auto atTensor##__MAYBE_WRAPPER = ::impl::aten::buildATenList(diopiTensors, numTensors);                                             \
    c10::List<c10::optional<at::Tensor>> atTensor;                                                                                      \
    atTensor.reserve(numTensors);                                                                                                       \
    std::transform(atTensor##__MAYBE_WRAPPER.begin(), atTensor##__MAYBE_WRAPPER.end(), std::back_inserter(atTensor), [](auto& tensor) { \
        return tensor.defined() ? c10::optional<at::Tensor>(tensor) : c10::nullopt;                                                     \
    });

}  // namespace impl::aten
