#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

// NOLINTBEGIN(readability-identifier-naming)

namespace impl {
namespace aten {
namespace utils {

namespace detail {

inline void noopDelete(void*) {}

}  // namespace detail

/// Provides a fluent API to construct tensors from external data.
///
/// The fluent API can be used instead of `from_blob` functions in case the
/// required set of parameters does not align with the existing overloads.
///
///     at::Tensor tensor = at::for_blob(data, sizes)
///             .strides(strides)
///             .context(context, [](void *ctx) { delete static_cast<Ctx*>(ctx); })
///             .options(...)
///             .make_tensor();
///
class TensorMakerXpuCompat {
    friend TensorMakerXpuCompat for_blob_xpu_compat(void* data, at::IntArrayRef sizes) noexcept;

public:
    using ContextDeleter = c10::DeleterFnPtr;

    TensorMakerXpuCompat& strides(at::OptionalIntArrayRef value) noexcept {
        strides_ = value;

        return *this;
    }

    TensorMakerXpuCompat& storage_offset(c10::optional<int64_t> value) noexcept {
        storage_offset_ = value;

        return *this;
    }

    TensorMakerXpuCompat& deleter(std::function<void(void*)> value) noexcept {
        deleter_ = std::move(value);

        return *this;
    }

    TensorMakerXpuCompat& context(void* value, ContextDeleter deleter = nullptr) noexcept {
        ctx_ = std::unique_ptr<void, ContextDeleter>{value, deleter != nullptr ? deleter : detail::noopDelete};

        return *this;
    }

    TensorMakerXpuCompat& target_device(c10::optional<c10::Device> value) noexcept {
        device_ = value;

        return *this;
    }

    TensorMakerXpuCompat& options(c10::TensorOptions value) noexcept {
        opts_ = value;

        return *this;
    }

    at::Tensor make_tensor();

private:
    explicit TensorMakerXpuCompat(void* data, at::IntArrayRef sizes) noexcept : data_{data}, sizes_{sizes} {}

    std::size_t computeStorageSize() const noexcept;

    c10::DataPtr makeDataPtrFromDeleter() const;

    c10::DataPtr makeDataPtrFromContext() noexcept;

    at::IntArrayRef makeTempSizes() const noexcept;

    void* data_;
    at::IntArrayRef sizes_;
    c10::OptionalIntArrayRef strides_{};
    c10::optional<int64_t> storage_offset_{};
    std::function<void(void*)> deleter_{};
    std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
    c10::optional<c10::Device> device_{};
    c10::TensorOptions opts_{};
};

inline TensorMakerXpuCompat for_blob_xpu_compat(void* data, at::IntArrayRef sizes) noexcept { return TensorMakerXpuCompat{data, sizes}; }

inline at::Tensor from_blob_xpu_compat(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, const std::function<void(void*)>& deleter,
                                       const c10::TensorOptions& options = {}, const c10::optional<c10::Device> target_device = c10::nullopt) {
    return for_blob_xpu_compat(data, sizes).strides(strides).deleter(deleter).options(options).target_device(target_device).make_tensor();
}

inline at::Tensor from_blob_xpu_compat(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, int64_t storage_offset,
                                       const std::function<void(void*)>& deleter, const c10::TensorOptions& options = {},
                                       const c10::optional<c10::Device> target_device = c10::nullopt) {
    return for_blob_xpu_compat(data, sizes)
        .strides(strides)
        .storage_offset(storage_offset)
        .deleter(deleter)
        .options(options)
        .target_device(target_device)
        .make_tensor();
}

inline at::Tensor from_blob_xpu_compat(void* data, at::IntArrayRef sizes, const std::function<void(void*)>& deleter, const c10::TensorOptions& options = {}) {
    return for_blob_xpu_compat(data, sizes).deleter(deleter).options(options).make_tensor();
}

inline at::Tensor from_blob_xpu_compat(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, const c10::TensorOptions& options = {}) {
    return for_blob_xpu_compat(data, sizes).strides(strides).options(options).make_tensor();
}

inline at::Tensor from_blob_xpu_compat(void* data, at::IntArrayRef sizes, const c10::TensorOptions& options = {}) {
    return for_blob_xpu_compat(data, sizes).options(options).make_tensor();
}

}  // namespace utils
}  // namespace aten
}  // namespace impl

// NOLINTEND(readability-identifier-naming)
