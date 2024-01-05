/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include "helper.hpp"

#include <ATen/cuda/EmptyTensor.h>

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
            return {};
    }
}

diopiDtype_t getDIOPITensorType(const at::Tensor& input) {
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

namespace {

template <diopiDevice_t>
class BuildATenDeviceImpl {};

template <>
class BuildATenDeviceImpl<diopi_host> {
public:
    static void lazyInitDevice() {}
    static at::Device device(diopiConstTensorHandle_t /*unused*/) { return {at::DeviceType::CPU}; }
    static at::Tensor empty(at::IntArrayRef size, at::ScalarType dtype, at::Device /*unused*/) {
        return at::detail::empty_cpu(size, dtype, /*pin_memory=*/false, /*memory_format_opt=*/c10::nullopt);
    }
};

template <>
class BuildATenDeviceImpl<diopi_device> {
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
at::Tensor buildATenImpl(diopiConstTensorHandle_t tensor) {
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

at::Tensor buildATen(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) {
        return at::Tensor();
    }

    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    switch (device) {
        case diopi_host:
            return buildATenImpl<BuildATenDeviceImpl<diopi_host>>(tensor);
        case diopi_device:
            return buildATenImpl<BuildATenDeviceImpl<diopi_device>>(tensor);
        default:
            TORCH_CHECK(false, "Invalid device type encountered in buildATen: ", device);
            return {};
    }
}

at::Scalar buildAtScalar(const diopiScalar_t* scalar) {
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

void buildDiopiTensor(diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t* out) {
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
at::Generator buildGenerator(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t generator) {
    auto gen = at::cuda::detail::createCUDAGenerator();
    diopiTensorHandle_t state_handle = nullptr;
    diopiGeneratorGetState(ctx, generator, &state_handle);
    auto state = impl::aten::buildATen(state_handle);
    {
        std::lock_guard<std::mutex> lock(gen.mutex());
        gen.set_state(state);
    }
    return gen;
}

void updateGeneratorHandleState(diopiContextHandle_t ctx, at::Generator& cuda_gen, diopiGeneratorHandle_t generator) {
    at::Tensor new_state;
    {
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        new_state = cuda_gen.get_state();
    }
    diopiTensorHandle_t new_state_handle = nullptr;
    buildDiopiTensor(ctx, new_state, &new_state_handle);
    diopiGeneratorSetState(generator, new_state_handle);
}

at::Tensor nllLossNdBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight, int64_t reduction,
                             int64_t ignore_index) {
    auto atWeight = buildATen(weight);

    /*
     * A tensor representing the sum of weights for each element considered in the NLL loss computation.
     * In case a weight tensor is provided, total_weight represents the sum of weights for all the non-ignored indices in the target tensor.
     * When no weight tensor is provided, total_weight corresponds to the count of all non-ignored indices.
     */
    at::Tensor atTotalWeight;
    // Flatten the target tensor for easier processing
    auto flatTarget = atTarget.view(-1);

    // Create a mask corresponding to ignore_index if it's provided
    auto mask = (ignore_index >= 0) ? (flatTarget != ignore_index) : at::ones(flatTarget.sizes(), flatTarget.options()).to(at::kBool);

    if (atWeight.defined()) {
        // Filter out the targets using the mask and compute total weight using index_select
        atTotalWeight = atWeight.index_select(0, flatTarget.masked_select(mask)).sum();
    } else {
        // If weight is not defined, compute total weight by counting the valid targets
        atTotalWeight = at::scalar_tensor(mask.sum().item<float>(), atInput.options());
    }

    auto dim = atInput.dim();
    if (dim >= 3 && dim != 4) {
        auto n = atInput.size(0);
        auto c = atInput.size(1);
        int64_t inputLastSize = 1;
        int64_t targetLastSize = 1;
        for (int i = 2; i < atInput.dim(); ++i) {
            inputLastSize *= atInput.size(i);
        }
        for (int i = 1; i < atTarget.dim(); ++i) {
            targetLastSize *= atTarget.size(i);
        }
        std::vector<int64_t> inputShape = {n, c, 1, inputLastSize};
        std::vector<int64_t> targetShape = {n, 1, targetLastSize};
        atInput = atInput.reshape(inputShape);
        atTarget = atTarget.reshape(targetShape);
        if (0 == reduction) {
            atGradOutput = atGradOutput.reshape(targetShape);
        }
    }
    at::Tensor atGradInput;
    if (dim >= 3) {
        atGradInput = at::nll_loss2d_backward(atGradOutput, atInput, atTarget, atWeight, reduction, ignore_index, atTotalWeight);
    } else {
        atGradInput = at::nll_loss_backward(atGradOutput, atInput, atTarget, atWeight, reduction, ignore_index, atTotalWeight);
    }
    return atGradInput;
}

at::Tensor crossEntropyLossProbTargetBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight,
                                              int64_t reduction, double label_smoothing) {
    auto atLogSoftmaxOutput = at::log_softmax(atInput, 1, atInput.scalar_type());
    at::Tensor atGradInput;
    const auto n_classes = atInput.size(1);
    if (label_smoothing > 0.0) {
        TORCH_CHECK(label_smoothing <= 1.0, "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);
        atTarget = atTarget * (1 - label_smoothing) + label_smoothing / n_classes;
    }
    std::vector<int64_t> expand_shape;
    for (int i = 0; i < atInput.dim(); ++i) {
        expand_shape.push_back(atInput.size(i));
    }
    at::IntArrayRef shape(expand_shape.data(), expand_shape.size());
    if (weight) {
        auto atWeight = buildATen(weight);
        std::vector<int64_t> weight_broadcast_shape(atInput.dim(), 1);
        weight_broadcast_shape[1] = atWeight.size(0);
        atWeight = atWeight.view(weight_broadcast_shape);
        switch (reduction) {
            case at::Reduction::Mean:
                atGradOutput = atGradOutput.expand(shape);
                atGradInput = -(atGradOutput * atTarget * atWeight) / (atInput.numel() / atInput.size(1));
                break;
            case at::Reduction::Sum:
                atGradOutput = atGradOutput.expand(shape);
                atGradInput = -(atGradOutput * atTarget * atWeight);
                break;
            case at::Reduction::None:
                atGradOutput = atGradOutput.unsqueeze(1).expand(shape);
                atGradInput = -(atGradOutput * atTarget * atWeight);
                break;
            default:
                TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
        }
    } else {
        switch (reduction) {
            case at::Reduction::Mean:
                atGradOutput = atGradOutput.expand(shape);
                atGradInput = -(atGradOutput * atTarget) / (atInput.numel() / atInput.size(1));
                break;
            case at::Reduction::Sum:
                atGradOutput = atGradOutput.expand(shape);
                atGradInput = -(atGradOutput * atTarget);
                break;
            case at::Reduction::None:
                atGradOutput = atGradOutput.unsqueeze(1).expand(shape);
                atGradInput = -(atGradOutput * atTarget);
                break;
            default:
                TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
        }
    }
    auto atGradInputFinal = at::_log_softmax_backward_data(atGradInput, atLogSoftmaxOutput, 1, atLogSoftmaxOutput.scalar_type());
    return atGradInputFinal;
}

at::Tensor crossEntropyLossLabelSmoothingBackward(at::Tensor& atInput, at::Tensor& atGradOutput, at::Tensor& atTarget, diopiConstTensorHandle_t weight,
                                                  int64_t reduction, int64_t ignore_index, double label_smoothing) {
    auto atLogSoftmaxOutput = at::log_softmax(atInput, 1, atInput.scalar_type());
    const auto n_classes = atInput.size(1);
    auto atNlllossGrad = atGradOutput * (1 - label_smoothing);
    auto atSmoothlossGrad = atGradOutput * (label_smoothing / n_classes);
    at::Tensor atGradInput;
    std::vector<int64_t> expand_shape;
    for (int i = 0; i < atInput.dim(); ++i) {
        if (i != 1) {
            expand_shape.push_back(atInput.size(i));
        }
    }
    at::IntArrayRef shape(expand_shape.data(), expand_shape.size());
    switch (reduction) {
        case at::Reduction::Mean:
            if (weight) {
                // loss is normalized by the weights to be consistent with nll_loss_nd
                auto atWeight = buildATen(weight);
                atGradInput = atSmoothlossGrad.expand(shape) / atWeight.gather(0, atTarget.flatten()).sum();
            } else {
                float num = 1.;
                for (int i = 0; i < expand_shape.size(); ++i) {
                    num *= expand_shape[i];
                }
                atGradInput = atSmoothlossGrad.expand(shape) / num;
            }
            break;
        case at::Reduction::Sum:
            atGradInput = atSmoothlossGrad.expand(shape);
            break;
        case at::Reduction::None:
            atGradInput = atSmoothlossGrad;
            break;
        default:
            TORCH_CHECK(false, "Invalid reduction type encountered in cross_entropy: ", reduction);
    }
    atGradInput = atGradInput.clone();
    if (ignore_index >= 0) {
        atGradInput.index_put_({atTarget == ignore_index}, 0.0);
    }
    std::vector<int64_t> final_expand_shape;
    for (int i = 0; i < atInput.dim(); ++i) {
        final_expand_shape.push_back(atInput.size(i));
    }
    at::IntArrayRef final_shape(final_expand_shape.data(), final_expand_shape.size());
    if (weight) {
        auto atWeight = buildATen(weight);
        std::vector<int64_t> weight_broadcast_shape(atInput.dim(), 1);
        weight_broadcast_shape[1] = atWeight.size(0);
        atWeight = atWeight.view(weight_broadcast_shape);
        atGradInput = -(atGradInput.unsqueeze(1).expand(final_shape) * atWeight);
    } else {
        atGradInput = -atGradInput.unsqueeze(1).expand(final_shape);
    }
    auto atGradInput2 = nllLossNdBackward(atLogSoftmaxOutput, atNlllossGrad, atTarget, weight, reduction, ignore_index);
    atGradInput = atGradInput.clone();
    atGradInput += atGradInput2;
    atLogSoftmaxOutput = at::log_softmax(atInput, 1, atInput.scalar_type());
    auto atGradInputFinal = at::_log_softmax_backward_data(atGradInput, atLogSoftmaxOutput, 1, atLogSoftmaxOutput.scalar_type());
    return atGradInputFinal;
}

}  // namespace aten

}  // namespace impl
