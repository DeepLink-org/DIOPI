/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "ascend_tensor.hpp"

#include "common/tensor_utils.hpp"

namespace impl {
namespace ascend {
AscendTensor::AscendTensor(const diopiTensorHandle_t& tensor) : tensor_(tensor) {
    if (tensor_ != nullptr) {
        diopiSize_t diopiShape;
        diopiGetTensorShape(tensor_, &diopiShape);
        std::vector<int64_t> shapeTmp(diopiShape.data, diopiShape.data + diopiShape.len);
        shape_ = std::move(shapeTmp);

        diopiSize_t diopiStride;
        diopiGetTensorStride(tensor_, &diopiStride);
        std::vector<int64_t> strideTmp(diopiStride.data, diopiStride.data + diopiStride.len);
        stride_ = std::move(strideTmp);
        ASCEND_CHECK_ABORT(stride_.size() == shape_.size(), "stride_.size() == shape_.size() check failed");

        diopiDtype_t diopiDtype;
        diopiGetTensorDtype(tensor_, &diopiDtype);
        dtype_ = diopiDtype;
    }
}

diopiDevice_t AscendTensor::device() const {
    ASCEND_CHECK_NULLPTR_ABORT(tensor_);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor_, &device);
    return device;
}

diopiDtype_t AscendTensor::dtype() const {
    ASCEND_CHECK_NULLPTR_ABORT(tensor_);
    return dtype_;
}

int64_t AscendTensor::numel() const {
    ASCEND_CHECK_NULLPTR_ABORT(tensor_);
    int64_t numel;
    diopiGetTensorNumel(tensor_, &numel);
    return numel;
}
int64_t AscendTensor::elemsize() const {
    ASCEND_CHECK_NULLPTR_ABORT(tensor_);
    int64_t elemsize;
    diopiGetTensorElemSize(tensor_, &elemsize);
    return elemsize;
}

bool AscendTensor::isContiguous(diopiMemoryFormat_t format) const {
    if (!defined()) {
        return true;
    }
    int64_t stride = 1;
    int64_t dim = this->dim();
    auto strides = this->stride();
    auto shape = this->shape();

    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }

    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
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
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
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
    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
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
    }
    return true;
}

AscendTensor& AscendTensor::asStrided(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
    this->shape_ = shape;
    this->stride_ = stride;
    return *this;
}

AscendTensor& AscendTensor::unsqueeze(int dim) {
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

AscendTensor& AscendTensor::view(const std::vector<int64_t> shape) {
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

void* AscendTensor::data() {
    void* p = nullptr;
    diopiGetTensorData(tensor_, &p);
    return p;
}
const void* AscendTensor::data() const {
    const void* p = nullptr;
    diopiGetTensorDataConst(tensor_, &p);
    return p;
}

diopiTensorHandle_t AscendTensor::tensorHandle() {
    if (this->defined()) {
        ASCEND_CHECK_ABORT(this->device() == diopiDevice_t::diopi_device, "%s", "tensor_ is not on camb device.");
    }
    return tensor_;
}

// diopiConstTensorHandle_t AscendTensor::tensorHandle() const {
//     if (this->defined()) {
//         ASCEND_CHECK_ABORT(this->device() == diopiDevice_t::diopi_device, "%s", "tensor_ is not on camb device.");
//     }
//     return tensor_;
// }

std::vector<int64_t> AscendTensor::getBaseShape() const {
    std::vector<int64_t> baseShapeVec;
    if (isContiguous()) {
        if (dim() > 0) {
            baseShapeVec.resize(dim());
            for (int64_t i = 0; i < dim(); i++) {
                baseShapeVec[i] = shape(i);
            }
        } else {
            baseShapeVec.push_back(1);
        }

    } else {
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < dim(); i++) {
            if (stride(i) > maxStride) {
                maxStride = stride(i);
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            baseShapeVec.push_back(shape(maxIdx) * maxStride);
        } else {
            baseShapeVec.push_back(1);
        }
    }
    return baseShapeVec;
}

int64_t AscendTensor::getBaseBufferSize() const {
    if (isContiguous()) {
        if (dim() > 0) {
            return this->numel() * this->elemsize();
        } else {
            return this->elemsize();
        }
    } else {
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < dim(); i++) {
            if (stride(i) > maxStride) {
                maxStride = stride(i);
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            return shape(maxIdx) * maxStride * this->elemsize();
        } else {
            return this->elemsize();
        }
    }
}

AscendTensor createAscendTensor(diopiContextHandle_t ctx, const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype, const diopiDevice_t dev) {
    diopiTensorHandle_t tensor = nullptr;
    diopiRequireTensor(ctx, &tensor, size, stride, dtype, dev);
    return AscendTensor(tensor);
}

AscendTensor createAscendTensor(diopiContextHandle_t ctx, const std::vector<int64_t>& size, const diopiSize_t* stride, const diopiDtype_t dtype, const diopiDevice_t dev) {
    diopiSize_t shape{size.data(), size.size()};
    return createAscendTensor(ctx, &shape, stride, dtype, dev);
}

}  // namespace ascend
}  // namespace impl
