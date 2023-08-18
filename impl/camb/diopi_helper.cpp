/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "diopi_helper.hpp"

namespace impl {
namespace camb {

DiopiTensor::DiopiTensor(const diopiTensorHandle_t& tensor) : tensor_(tensor) {
    if (tensor_ != nullptr) {
        diopiSize_t diopiShape;
        diopiSize_t diopiStride;
        diopiDtype_t diopiDtype;
        diopiGetTensorShape(tensor_, &diopiShape);
        std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
        diopiGetTensorStride(tensor_, &diopiStride);
        std::vector<int64_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
        diopiGetTensorDtype(tensor_, &diopiDtype);
        shape_ = std::move(shapeTmp);
        stride_ = std::move(strideTmp);
        dtype_ = diopiDtype;
    }
}

diopiDevice_t DiopiTensor::device() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor_, &device);
    return device;
}

diopiDtype_t DiopiTensor::dtype() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    return dtype_;
}

DiopiTensor DiopiTensor::viewAsComplex() const {
    int64_t lastDim = size(-1);
    DIOPI_CHECK_ABORT(2 == lastDim, "last dim of tensor must be 2 when view as complex");
    diopiDtype_t complexDtype = DiopiDataType::realDtype2Complex(dtype());
    std::vector<int64_t> complexShape(shape().begin(), shape().end() - 1);
    std::vector<int64_t> complexStride(stride().begin(), stride().end() - 1);
    for (auto& i : complexStride) {
        i /= 2;
    }
    DiopiTensor complexTensor(tensor_);
    complexTensor.asStrided(complexShape, complexStride).setDtype(complexDtype);
    return complexTensor;
}

DiopiTensor DiopiTensor::viewAsReal() const {
    diopiDtype_t realDtype = DiopiDataType::complexDtype2Real(dtype());
    std::vector<int64_t> realShape(shape());
    realShape.push_back(2);
    std::vector<int64_t> realStride(stride());
    for (auto& i : realStride) {
        i *= 2;
    }
    realStride.push_back(1);
    DiopiTensor realTensor(tensor_);
    realTensor.asStrided(realShape, realStride).setDtype(realDtype);
    return realTensor;
}

const std::vector<int64_t>& DiopiTensor::shape() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    return shape_;
}

const std::vector<int64_t>& DiopiTensor::stride() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    return stride_;
}

int64_t DiopiTensor::numel() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    int64_t numel;
    diopiGetTensorNumel(tensor_, &numel);
    return numel;
}
int64_t DiopiTensor::elemsize() const {
    DIOPI_CHECK_NULLPTR_ABORT(tensor_);
    int64_t elemsize;
    diopiGetTensorElemSize(tensor_, &elemsize);
    return elemsize;
}

DiopiTensor DiopiTensor::contiguous(diopiContextHandle_t ctx, MemoryFormat format) {
    /* DEPRECATED AND WILL BE REMOVED */
    if (this->isContiguous(format)) return *this;
    int64_t dim = this->dim();
    std::vector<int64_t> strides(dim);
    int64_t stride = 1;
    if (format == MemoryFormat::Contiguous) {
        for (size_t i = dim; i > 0; --i) {
            strides[i - 1] = stride;
            if (shape_[i - 1] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= shape_[i - 1];
            }
        }
    } else if (format == MemoryFormat::ChannelsLast) {
        DIOPI_CHECK_ABORT(this->shape().size() == 4, "%s", "tensor size should be 4");
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = stride;
            if (shape_[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= shape_[k];
            }
        }
    } else if (format == MemoryFormat::ChannelsLast3d) {
        DIOPI_CHECK_ABORT(this->shape().size() == 5, "%s", "tensor size should be 5");
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = stride;
            if (shape_[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= shape_[k];
            }
        }
    }
    diopiSize_t strideDiopi{strides.data(), static_cast<int64_t>(strides.size())};
    diopiSize_t shapeDiopi{this->shape().data(), static_cast<int64_t>(this->shape().size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &shapeDiopi, &strideDiopi, this->dtype(), this->device());
    return DiopiTensor(tensor);
}

bool DiopiTensor::isContiguous(MemoryFormat format) const {
    if (!defined()) {
        return true;
    }
    int64_t stride = 1;
    int64_t dim = this->dim();
    auto strides = this->stride();
    auto shape = this->shape();

    if (format == MemoryFormat::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == MemoryFormat::ChannelsLast1d) {
        if (strides.size() != 3) {
            return false;
        }
        for (auto& i : {1, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }

    } else if (format == MemoryFormat::ChannelsLast) {
        if (strides.size() != 4) return false;
        for (auto& i : {1, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                // shape_d != 1 help dealing with shape like [2, 2048, 1, 1]
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == MemoryFormat::ChannelsLast3d) {
        if (strides.size() != 5) return false;
        for (auto& i : {1, 4, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    }
    return true;
}

DiopiTensor& DiopiTensor::asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
    this->shape_ = shape;
    this->stride_ = stride;
    return *this;
}

DiopiTensor& DiopiTensor::unsqueeze(int dim) {
    // Note: `channels_last` tensor uses this will become uncontiguous
    // which is same with pytorch
    auto shape = this->shape();
    auto strides = this->stride();
    int64_t newStride = dim >= this->dim() ? 1 : shape[dim] * strides[dim];
    std::vector<int64_t> newShape(shape.begin(), shape.end());
    std::vector<int64_t> newStrides(strides.begin(), strides.end());

    newShape.insert(newShape.begin() + dim, 1);
    newStrides.insert(newStrides.begin() + dim, newStride);
    this->asStrided(newShape, newStrides);
    return *this;
}

DiopiTensor& DiopiTensor::view(const std::vector<int64_t> shape) {
    // must be contiguous
    std::vector<int64_t> stride(shape.size());
    this->shape_ = shape;
    stride[shape.size() - 1] = 1;
    for (int j = shape_.size() - 2; j >= 0; j--) {
        stride[j] = stride[j + 1] * shape[j + 1];
    }
    this->stride_ = stride;
    return *this;
}

void* DiopiTensor::data() {
    void* p = nullptr;
    diopiGetTensorData(tensor_, &p);
    return p;
}
const void* DiopiTensor::data() const {
    const void* p = nullptr;
    diopiGetTensorDataConst(tensor_, &p);
    return p;
}

MemoryFormat DiopiTensor::suggestMemoryFormat() {
    // TODO(waiting for dispatch): Performance can be improved by dividing is_contiguous into several funcs
    if (this->isContiguous(MemoryFormat::Contiguous)) {
        return MemoryFormat::Contiguous;
    } else if (this->isContiguous(MemoryFormat::ChannelsLast)) {
        return MemoryFormat::ChannelsLast;
    } else {
        return MemoryFormat::ChannelsLast3d;
    }
}
diopiTensorHandle_t DiopiTensor::tensorHandle() {
    if (this->defined()){
        DIOPI_CHECK_ABORT(this->device() == diopiDevice_t::diopi_device, "%s", "tensor_ is not on camb device.");
    }
    return tensor_;
}

diopiConstTensorHandle_t DiopiTensor::tensorHandle() const {
    if (this->defined()){
        DIOPI_CHECK_ABORT(this->device() == diopiDevice_t::diopi_device, "%s", "tensor_ is not on camb device.");
    }
    return tensor_;
}

DiopiTensor makeTensor(diopiContextHandle_t ctx, const diopiScalar_t* pScalar) {
    diopiTensorHandle_t tensor = nullptr;
    std::vector<int64_t> shape{1};
    diopiSize_t size{shape.data(), 1};
    diopiRequireTensor(ctx, &tensor, &size, nullptr, pScalar->stype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor ones(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    diopiScalar_t scalar = constructDiopiScalarT(dtype, 1);
    diopiFill(ctx, tensor, &scalar);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const diopiSize_t& size, diopiDtype_t dtype) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const std::vector<int64_t>& stride, diopiDtype_t dtype) {
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiSize_t strideTmp{stride.data(), static_cast<int64_t>(stride.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, &strideTmp, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype) {
    diopiSize_t sizeTmp{size.data(), static_cast<int64_t>(size.size())};
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, &sizeTmp, nullptr, dtype, diopi_device);
    return DiopiTensor(tensor);
}

DiopiTensor requiresTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, diopiDtype_t dtype, MemoryFormat memoryFormat) {
    int64_t dim = size.size();
    std::vector<int64_t> strides(dim);
    int64_t stride = 1;
    if (memoryFormat == MemoryFormat::Contiguous) {
        for (size_t i = dim; i > 0; --i) {
            strides[i - 1] = stride;
            if (size[i - 1] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[i - 1];
            }
        }
    } else if (memoryFormat == MemoryFormat::ChannelsLast1d) {
        DIOPI_CHECK_ABORT(size.size() == 3, "%s", "tensor size should be 3");
        for (auto& k : {1, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }

    } else if (memoryFormat == MemoryFormat::ChannelsLast) {
        DIOPI_CHECK_ABORT(size.size() == 4, "%s", "tensor size should be 4");
        // constant array is used here to let
        // compiler fully unroll the loop to get better performance
        for (auto& k : {1, 3, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }
    } else if (memoryFormat == MemoryFormat::ChannelsLast3d) {
        DIOPI_CHECK_ABORT(size.size() == 5, "%s", "tensor size should be 5");
        for (auto& k : {1, 4, 3, 2, 0}) {
            strides[k] = stride;
            if (size[k] == 0) {
                continue;
            }
            if (stride != -1) {
                stride *= size[k];
            }
        }
    } else {
        DIOPI_CHECK_ABORT(false, "memory format not support");
    }
    return requiresTensor(ctx, size, strides, dtype);
}

}  // namespace camb

}  // namespace impl
