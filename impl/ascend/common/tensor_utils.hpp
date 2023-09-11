/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_TENSOR_UTILS_HPP_
#define IMPL_ASCEND_COMMON_TENSOR_UTILS_HPP_

#include <acl/acl.h>
#include <diopi/diopirt.h>

#include <algorithm>
#include <array>
#include <vector>

#include "../error.hpp"
#include "debug.hpp"

namespace impl {
namespace ascend {

// TODO: 调整下面3个函数的位置，现在位置太差了。
aclDataType getAclDataType(diopiDtype_t type);
aclDataType getAclDataType(diopiConstTensorHandle_t th);

inline aclFormat getAclDataFormat(diopiConstTensorHandle_t th) {
    diopiSize_t shape;
    diopiSize_t stride;
    diopiGetTensorShape(th, &shape);
    diopiGetTensorStride(th, &stride);
    ASCEND_CHECK_ABORT(stride.len == shape.len, "stride.len == shape.len check failed");
    if (shape.len == 4) {
        std::array<int64_t, 4> thStride{stride.data[0], stride.data[1], stride.data[2], stride.data[3]};
        {
            std::array<int64_t, 4> nchwStride;
            int st = 1;
            for (auto k : {3, 2, 1, 0}) {
                nchwStride[k] = st;
                if (shape.data[k] == 0) continue;
                if (shape.data[k] == -1) st = -1;
                if (st != -1) st *= shape.data[k];
            }
            if (thStride == nchwStride) {
                return ACL_FORMAT_NCHW;
            }
        }
        std::array<int64_t, 4> nhwcStride;
        int st = 1;
        for (auto k : {1, 3, 2, 0}) {
            nhwcStride[k] = st;
            if (shape.data[k] == 0) continue;
            if (shape.data[k] == -1) st = -1;
            if (st != -1) st *= shape.data[k];
        }
        if (thStride == nhwcStride) {
            return ACL_FORMAT_NHWC;
        }
        warning("[OLD]Acl only support NCHW or NHWC format! but get %s", dumpTensor(th).c_str());
    }
    return ACL_FORMAT_ND;
}

inline bool isIntegralType(const diopiDtype_t& type) { return type < 8; }

inline bool isIntegralTypeWithBool(const diopiDtype_t& type) { return type < 8 || type == 11; }

inline bool isFloatingType(const diopiDtype_t& type) { return (type <= 10 && type >= 8) || type == 12 || type == 13; }

template <typename T>
T getValue(const diopiScalar_t* scalar) {
    ASCEND_CHECK_ABORT(scalar != nullptr, "input should not be nullptr");
    if (isIntegralTypeWithBool(scalar->stype)) {
        return static_cast<T>(scalar->ival);
    } else {
        return static_cast<T>(scalar->fval);
    }
}

diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* out, float val);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out);

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype);

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype);

diopiTensorHandle_t hostToDevice(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

inline std::vector<int64_t> calcStrides(int ndims, diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    std::vector<int64_t> strides;
    strides.resize(ndims);
    int64_t st = 1;
    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = ndims; i > 0; --i) {
            strides[i - 1] = st;
            if (size.data[i - 1] == 0) continue;
            if (size.data[i - 1] == -1) st = -1;
            if (st != -1) st *= size.data[i - 1];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) st *= size.data[k];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }

    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        for (auto k : {1, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }
    } else {
        // PARROTS_THROW(InvalidArgs) <<
        //         "Invalid MemoryFormat " << memoryFormatName(format);
    }
    return strides;
}

inline bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format = diopiMemoryFormat_t::ChannelsLast) {
    diopiSize_t shape, stride;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    if (shape.len != 4) return false;
    int64_t totalSize = 1;
    for (int64_t i = 0; i < shape.len; ++i) {
        totalSize *= shape.data[i];
    }
    if (totalSize == 0) return false;
    if (stride.data[0] == stride.data[1]) return false;
    if (checkContiguous) {
        auto realStride = calcStrides(shape.len, shape, format);
        for (int i = 0; i < stride.len; ++i) {
            if (i >= realStride.size() || realStride[i] != stride.data[i]) {
                return false;
            }
        }
        return true;
    } else {
        int64_t st = 1;
        std::vector<int> orders;
        if (format == diopiMemoryFormat_t::ChannelsLast)
            orders = {1, 3, 2, 0};
        else if (format == diopiMemoryFormat_t::ChannelsLast3d)
            orders = {1, 4, 3, 2, 0};
        for (auto k : orders) {
            if (stride.data[k] < st) return false;
            st = stride.data[k] * shape.data[k];
        }
        return true;
    }
}

inline diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch = false) {
    return isLikeChannelsLast(tensor, exactMatch)
               ? diopiMemoryFormat_t::ChannelsLast
               : (isLikeChannelsLast(tensor, exactMatch, diopiMemoryFormat_t::ChannelsLast3d) ? diopiMemoryFormat_t::ChannelsLast3d
                                                                                              : diopiMemoryFormat_t::Contiguous);
}

bool isContiguous(diopiConstTensorHandle_t tensor, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

diopiTensorHandle_t clone(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiDtype_t dtype);

int64_t getBaseBufferSize(diopiConstTensorHandle_t src);

std::vector<int64_t> getBaseShape(diopiConstTensorHandle_t src);

diopiSize_t vectorToDiopiSize(std::vector<int64_t>& sizeVec);

diopiSize_t arrayToDiopiSize(int64_t* data, int64_t len);

diopiError_t contiguous(diopiContextHandle_t ctx, AscendTensor& src);

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_COMMON_TENSOR_UTILS_HPP_
