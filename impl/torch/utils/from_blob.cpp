#include "from_blob.h"

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Utils.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// NOLINTBEGIN(readability-identifier-naming)

namespace impl {
namespace aten {
namespace utils {

at::Tensor TensorMakerXpuCompat::make_tensor() {
    at::check_size_nonnegative(sizes_);

    TORCH_CHECK_VALUE(!deleter_ || !ctx_, "The deleter and context arguments are mutually exclusive.");

    if (device_ == c10::nullopt) {
        auto device_type = opts_.device().type();
        if (device_type == c10::DeviceType::XPU) {
            device_ = at::globalContext().getDeviceFromPtr(data_, c10::DeviceType::CUDA);
            device_.emplace(c10::DeviceType::XPU, device_->index());
        } else {
            device_ = at::globalContext().getDeviceFromPtr(data_, device_type);
        }
    }

    if (opts_.device().has_index()) {
        TORCH_CHECK_VALUE(opts_.device() == *device_, "Specified device ", opts_.device(), " does not match device of data ", *device_);
    }

    std::size_t size_bytes = computeStorageSize();

    c10::DataPtr data_ptr{};
    if (deleter_) {
        data_ptr = makeDataPtrFromDeleter();
    } else {
        data_ptr = makeDataPtrFromContext();
    }

    c10::Storage storage{c10::Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr)};

    at::Tensor tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

    c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
    if (strides_) {
        tensor_impl->set_sizes_and_strides(sizes_, *strides_);
    } else {
        tensor_impl->set_sizes_contiguous(sizes_);
    }
    if (storage_offset_) {
        tensor_impl->set_storage_offset(*storage_offset_);
    }

    return tensor;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
std::size_t TensorMakerXpuCompat::computeStorageSize() const noexcept {
    std::size_t itemsize = opts_.dtype().itemsize();

    if (strides_) {
        auto storage_size = at::detail::computeStorageNbytes(sizes_, *strides_, itemsize);
        if (storage_offset_) {
            storage_size += storage_offset_.value();
        }
        return storage_size;
    }

    std::size_t size = 1;
    for (std::int64_t s : sizes_) {
        size *= static_cast<std::size_t>(s);
    }
    auto storage_size = size * itemsize;
    if (storage_offset_) {
        storage_size += storage_offset_.value();
    }
    return storage_size;
}

c10::DataPtr TensorMakerXpuCompat::makeDataPtrFromDeleter() const { return c10::InefficientStdFunctionContext::makeDataPtr(data_, deleter_, *device_); }

c10::DataPtr TensorMakerXpuCompat::makeDataPtrFromContext() noexcept { return c10::DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_}; }

at::IntArrayRef TensorMakerXpuCompat::makeTempSizes() const noexcept {
    static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
    if (opts_.has_memory_format()) {
        c10::MemoryFormat format = *opts_.memory_format_opt();
        if (format == c10::MemoryFormat::ChannelsLast) {
            return at::IntArrayRef(zeros, 4);
        }
        if (format == c10::MemoryFormat::ChannelsLast3d) {
            return at::IntArrayRef(zeros, 5);
        }
    }
    return at::IntArrayRef(zeros, 1);
}

}  // namespace utils
}  // namespace aten
}  // namespace impl

// NOLINTEND(readability-identifier-naming)
