#include "build_aten.hpp"

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <diopi/diopirt.h>

#include <utility>

#include "helper.hpp"

namespace impl::aten {

UnsafelyDeviceChangedTensorWrapper::UnsafelyDeviceChangedTensorWrapper(const at::Tensor& tensor) : at::Tensor(tensor) {
    if (!defined() || is_cpu()) {
        return;
    }
    saveForRevert_.emplace(unsafeGetTensorImpl(), device());
    // NOTE: CUDA allocators may have not been initialized if we were using DIPU allocators.
    //       We have to do this explicitly for potential allocations in op workspaces.
    at::globalContext().lazyInitCUDA();
    at::Device newDevice{at::DeviceType::CUDA, device().index()};
    setTensorImplDeviceUnsafe({unsafeGetTensorImpl(), newDevice});
}

UnsafelyDeviceChangedTensorWrapper::~UnsafelyDeviceChangedTensorWrapper() {
    if (saveForRevert_.has_value()) {
        setTensorImplDeviceUnsafe(*saveForRevert_);
    }
}

UnsafelyDeviceChangedTensorWrapper buildATenUnsafe(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) {
        return {};
    }
    auto& atTensor = *reinterpret_cast<at::Tensor*>(const_cast<diopiTensorHandle_t>(tensor));
    return UnsafelyDeviceChangedTensorWrapper::createFromTensor(atTensor);
}

void UnsafelyDeviceChangedTensorWrapper::setTensorImplDeviceUnsafe(const TensorImplAndDevice& tensorAndDevice) {
    const auto& [tensorImpl, device] = tensorAndDevice;
    auto& storage = const_cast<at::Storage&>(tensorImpl->unsafe_storage());
    auto& dataPtr = const_cast<at::DataPtr&>(storage.data_ptr());
    dataPtr.unsafe_set_device(device);
    tensorImpl->set_storage_keep_dtype(std::move(storage));
    tensorImpl->_change_backend_component_keys(device);
}

namespace {

template <diopiDevice_t>
class BuildATenDeviceApi {};

template <>
class BuildATenDeviceApi<diopi_host> {
public:
    static void lazyInitDevice() {}
    static at::Device device(diopiConstTensorHandle_t /*unused*/) { return {at::DeviceType::CPU}; }
    static at::Tensor empty(at::IntArrayRef size, at::ScalarType dtype, at::Device /*unused*/) {
        return at::detail::empty_cpu(size, dtype, /*pin_memory=*/false, /*memory_format_opt=*/c10::nullopt);
    }
};

template <>
class BuildATenDeviceApi<diopi_device> {
public:
    static void lazyInitDevice() { at::globalContext().lazyInitCUDA(); }
    static at::Device device(diopiConstTensorHandle_t tensor) {
        diopiDeviceIndex_t deviceIndex;
        diopiGetTensorDeviceIndex(tensor, &deviceIndex);
        return {at::DeviceType::CUDA, deviceIndex};
    }
    static at::Tensor empty(at::IntArrayRef size, at::ScalarType dtype, at::Device device) {
        return at::detail::empty_cuda(size, dtype, device, /*memory_format_opt=*/c10::nullopt);
    }
};

template <class DeviceImpl>
at::Tensor buildATenSafeImpl(diopiConstTensorHandle_t tensor) {
    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    at::IntArrayRef atSizes(shape.data, shape.len);

    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    auto atTypeMeta = getATenType(dtype);
    auto atDtype = atTypeMeta.toScalarType();

    auto atDevice = DeviceImpl::device(tensor);

    // NOTE: storage offset has been handled in `diopiGetTensorData`
    void* data = nullptr;
    diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);

    if (data == nullptr) {
        return DeviceImpl::empty(atSizes, atDtype, atDevice);
    }

    // NOTE: CUDA allocators may have not been initialized if we were using DIPU allocators.
    //       We have to do this explicitly for potential allocations in op workspaces.
    DeviceImpl::lazyInitDevice();

    // PERF: It would be faster if we can obtain and reuse the storage from tensor.
    //       However we cannot assume diopiTensorHandle_t to be a wrapper of at::Tensor.
    //       So we have to create a new storage (offset = 0) whose data_ptr points to
    //       the same address but with an empty dtor (to avoid double-free).

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    at::IntArrayRef atStrides(stride.data, stride.len);

    auto storageNBytes = at::detail::computeStorageNbytes(atSizes, atStrides, atTypeMeta.itemsize());

    // NOTE: in this way, data_ptr will have an empty destructor
    at::Storage storage{at::Storage::use_byte_size_t{}, storageNBytes, /*data_ptr=*/{data, atDevice}};

    auto dk = at::computeDispatchKey(atDtype, /*layout=*/c10::nullopt, atDevice);
    at::Tensor atTensor = at::detail::make_tensor<at::TensorImpl>(std::move(storage), dk, atTypeMeta);
    atTensor.unsafeGetTensorImpl()->set_sizes_and_strides(atSizes, atStrides);

    return atTensor;
}

}  // namespace

at::Tensor buildATenSafe(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) {
        return at::Tensor();
    }

    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    switch (device) {
        case diopi_host:
            return buildATenSafeImpl<BuildATenDeviceApi<diopi_host>>(tensor);
        case diopi_device:
            return buildATenSafeImpl<BuildATenDeviceApi<diopi_device>>(tensor);
        default:
            TORCH_CHECK(false, "Invalid device type encountered in buildATen: ", device);
            return {};
    }
}

}  // namespace impl::aten
