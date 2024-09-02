/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "ascend_tensor.hpp"

// #include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <numeric>
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

AscendTensor& AscendTensor::permute(std::vector<int64_t> dims) {
    ASCEND_CHECK_ABORT(this->dim() == dims.size(), "permute dims does not match the tensor dims.");

    std::vector<int64_t> newShape(dims.size(), 0);
    std::vector<int64_t> newStride(dims.size(), 0);

    for (size_t i = 0; i < dims.size(); i++) {
        newShape[i] = this->shape(dims[i]);
        newStride[i] = this->stride(dims[i]);
    }

    this->shape_ = newShape;
    this->stride_ = newStride;

    return *this;
}

AscendTensor& AscendTensor::expand(std::vector<int64_t> shape) {
    ASCEND_CHECK_ABORT(shape.size() >= this->dim(),
                       "the number of sizes provided[% ld] must be greater or eaqual to the number of dimensions of the tensor[% ld].",
                       shape.size(),
                       this->dim());

    // todo: dim() == 0
    int64_t expandDims = shape.size() - this->shape().size();
    std::vector<int64_t> tShapeExp(expandDims, 0);
    auto tShape = this->shape();
    tShapeExp.insert(tShapeExp.end(), tShape.begin(), tShape.end());
    std::vector<int64_t> newShape = shape;

    for (int64_t i = 0; i < newShape.size(); i++) {
        if (newShape[i] < 0 && i < expandDims) {
            ASCEND_CHECK_ABORT(false, "The expanded size of the tensor (%ld) isn't allowed in a leading, non-existing dimension %ld", newShape[i], i);
        }

        if (i >= expandDims) {
            if (newShape[i] == -1) {
                newShape[i] = tShapeExp[i];
            } else {
                ASCEND_CHECK_ABORT(tShapeExp[i] == 1 || newShape[i] == tShapeExp[i],
                                   "The expanded size of the tensor (%ld) must match the existing size (%ld) at non-singleton dimension %ld.",
                                   newShape[i],
                                   tShapeExp[i],
                                   i);
            }
        }
    }

    int64_t numElem = std::accumulate(newShape.begin(), newShape.end(), 1, std::multiplies<>());
    std::vector<int64_t> newStride(expandDims, 0);
    auto tStride = this->stride();
    newStride.insert(newStride.end(), tStride.begin(), tStride.end());
    for (int64_t i = expandDims; i < shape.size(); i++) {
        if (shape[i] == -1 || shape[i] == tShapeExp[i]) {
            continue;
        } else {
            newStride[i] = 0;
        }
    }

    this->numel_ = numElem;
    this->shape_ = newShape;
    this->stride_ = newStride;

    return *this;
}

AscendTensor& AscendTensor::resize(const std::vector<int64_t>& shape) {
    int64_t numElem = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t j = shape.size() - 2; j >= 0; j--) {
        stride[j] = stride[j + 1] * shape[j + 1];
    }

    this->numel_ = numElem;
    this->shape_ = shape;
    this->stride_ = stride;

    return *this;
}
AscendTensor& AscendTensor::select(int64_t dim, int64_t index) {
    auto shape = this->shape();
    auto stride = this->stride();

    ASCEND_CHECK_ABORT(dim >= 0 && dim < shape.size(), "selected dim [%ld] execeed the tensor dims [%ld].", dim, shape.size());

    if (dim < shape.size() - 1) {
        int64_t offset = dim * shape[dim] * stride[dim];
        this->storageOffset_ = offset;
    }
    this->numel_ /= shape[dim];

    shape.erase(shape.begin() + dim);
    stride.erase(stride.begin() + dim);
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

AscendTensor& AscendTensor::squeeze(int dim) {
    auto shape = this->shape();
    auto strides = this->stride();

    shape.erase(shape.begin() + dim);
    strides.erase(strides.begin() + dim);

    this->asStrided(shape, strides);
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

aclFormat inferAclDataFormat(int64_t dim, const int64_t* shape, const int64_t* stride) {
    static std::once_flag warningFlag;
    auto warnOnUnsupportedFormat = [dim, shape, stride](const char* file, int line, const char* func) {
        std::string msg = "Acl only support NCHW or NHWC format! but get shape = [";
        for (int64_t i = 0; i < dim; i++) {
            msg += std::to_string(shape[i]) + (i == dim - 1 ? "]" : ", ");
        }
        msg += ", stride = [";
        for (int64_t i = 0; i < dim; i++) {
            msg += std::to_string(stride[i]) + (i == dim - 1 ? "]" : ", ");
        }
        warning(file, line, func, msg.c_str());
    };
    if (dim == 5) {
        std::array<int64_t, 5> thStride{stride[0], stride[1], stride[2], stride[3], stride[4]};
        int st = 1;
        std::array<int64_t, 5> ncdhwStride;
        for (auto k : {4, 3, 2, 1, 0}) {
            ncdhwStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == ncdhwStride) {
            return ACL_FORMAT_NCDHW;
        }

        st = 1;
        std::array<int64_t, 5> ndhwcStride;
        for (auto k : {1, 4, 3, 2, 0}) {
            ndhwcStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == ndhwcStride) {
            return ACL_FORMAT_NDHWC;
        }
        std::call_once(warningFlag, warnOnUnsupportedFormat, __FILE__, __LINE__, __FUNCTION__);
    } else if (dim == 4) {
        std::array<int64_t, 4> thStride{stride[0], stride[1], stride[2], stride[3]};
        {
            std::array<int64_t, 4> nchwStride;
            int st = 1;
            for (auto k : {3, 2, 1, 0}) {
                nchwStride[k] = st;
                if (shape[k] == 0) continue;
                if (shape[k] == -1) st = -1;
                if (st != -1) st *= shape[k];
            }
            if (thStride == nchwStride) {
                return ACL_FORMAT_NCHW;
            }
        }
        std::array<int64_t, 4> nhwcStride;
        int st = 1;
        for (auto k : {1, 3, 2, 0}) {
            nhwcStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == nhwcStride) {
            return ACL_FORMAT_NHWC;
        }
        std::call_once(warningFlag, warnOnUnsupportedFormat, __FILE__, __LINE__, __FUNCTION__);
    } else if (dim == 3) {
        return ACL_FORMAT_NCL;
    }
    return ACL_FORMAT_ND;
}
}  // namespace ascend
}  // namespace impl
