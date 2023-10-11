/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "ascend_tensor.hpp"

#include <array>
#include <utility>

#include "common/debug.hpp"

namespace impl {
namespace ascend {

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

AscendTensor& AscendTensor::view(const std::vector<int64_t>& shape) {
    // must be contiguous
    ASCEND_CHECK_ABORT(this->isContiguous(), "now only contiguous tensor support view by shape.");
    std::vector<int64_t> stride(shape.size());
    this->shape_ = shape;
    stride[shape.size() - 1] = 1;
    for (int j = shape_.size() - 2; j >= 0; j--) {
        stride[j] = stride[j + 1] * shape[j + 1];
    }
    this->stride_ = stride;
    return *this;
}

const void* AscendTensor::data() const {
    const void* p = nullptr;
    diopiGetTensorDataConst(tensor_, &p);
    return p;
}

std::vector<int64_t> AscendTensor::getAclMemShape() const {
    std::vector<int64_t> baseShapeVec;
    if (this->isContiguous()) {
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

int64_t AscendTensor::getAclMemBufferSize() const {
    if (this->isContiguous()) {
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

aclFormat AscendTensor::getAclDataFormat() const {
    if (dim() == 4) {
        std::array<int64_t, 4> thStride{stride(0), stride(1), stride(2), stride(3)};
        {
            std::array<int64_t, 4> nchwStride;
            int st = 1;
            for (auto k : {3, 2, 1, 0}) {
                nchwStride[k] = st;
                if (shape(k) == 0) continue;
                if (shape(k) == -1) st = -1;
                if (st != -1) st *= shape(k);
            }
            if (thStride == nchwStride) {
                return ACL_FORMAT_NCHW;
            }
        }
        std::array<int64_t, 4> nhwcStride;
        int st = 1;
        for (auto k : {1, 3, 2, 0}) {
            nhwcStride[k] = st;
            if (shape(k) == 0) continue;
            if (shape(k) == -1) st = -1;
            if (st != -1) st *= shape(k);
        }
        if (thStride == nhwcStride) {
            return ACL_FORMAT_NHWC;
        }
        warning("getAclDataFormat error. Acl only support NCHW or NHWC format! but get %s", dumpTensor(tensor_).c_str());
    }
    return ACL_FORMAT_ND;
}

aclDataType AscendTensor::getAclDataType() const {
    switch (dtype_) {
        case diopi_dtype_float16:
            return ACL_FLOAT16;
        case diopi_dtype_float32:
            return ACL_FLOAT;
        case diopi_dtype_float64:
            return ACL_DOUBLE;
        case diopi_dtype_int8:
            return ACL_INT8;
        case diopi_dtype_uint8:
            return ACL_UINT8;
        case diopi_dtype_int16:
            return ACL_INT16;
        case diopi_dtype_uint16:
            return ACL_UINT16;
        case diopi_dtype_int32:
            return ACL_INT32;
        case diopi_dtype_uint32:
            return ACL_UINT32;
        case diopi_dtype_int64:
            return ACL_INT64;
        case diopi_dtype_uint64:
            return ACL_UINT64;
        case diopi_dtype_bool:
            return ACL_BOOL;
        case diopi_dtype_complex64:
            return ACL_COMPLEX64;
        case diopi_dtype_complex128:
            return ACL_COMPLEX128;
        default:
            return ACL_DT_UNDEFINED;
    }
}

}  // namespace ascend
}  // namespace impl
