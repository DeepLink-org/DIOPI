/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <vector>

#include "../common/debug.hpp"
#include "common.hpp"

namespace impl {
namespace camb {

static diopiError_t transpose(diopiContextHandle_t& ctx, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut,
                              std::vector<int32_t> order) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inDesc(in, layoutIn);
    CnnlTensorDesc outDesc(out, layoutOut);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? requiresBuffer(ctx, workspaceSize).data() : nullptr;
    DIOPI_CALL_CNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data(), workspacePtr, workspaceSize));
    return diopiSuccess;
}

// static diopiError_t calTensordiopiMemoryFormat_t(const DiopiTensor& tensor, diopiMemoryFormat_t& memoryFormatOut) {
//     if (tensor.isContiguous(diopiMemoryFormat_t::ChannelsLast)) {
//         memoryFormatOut = diopiMemoryFormat_t::ChannelsLast;
//     } else if (tensor.isContiguous(diopiMemoryFormat_t::ChannelsLast3d)) {
//         memoryFormatOut = diopiMemoryFormat_t::ChannelsLast3d;
//     } else if (tensor.isContiguous(diopiMemoryFormat_t::Contiguous)) {
//         memoryFormatOut = diopiMemoryFormat_t::Contiguous;
//     } else {
//         return diopiNoImplement;
//     }
//     return diopiSuccess;
// }

static diopiError_t calOrderAndSrcMemoryFormat(const DiopiTensor& src, diopiMemoryFormat_t destMemoryFormat, diopiMemoryFormat_t& srcMemoryFormatOut,
                                               std::vector<int32_t>& orderOut, std::vector<int32_t>& reverseOrder) {
    if (src.isContiguous(destMemoryFormat)) {
        srcMemoryFormatOut = destMemoryFormat;
        orderOut.reserve(src.dim());
        for (int i = 0; i < src.dim(); ++i) {
            orderOut[i] = i;
        }
        reverseOrder = orderOut;
        return diopiSuccess;
    }
    if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast1d) && destMemoryFormat == diopiMemoryFormat_t::Contiguous) {
        if (src.dim() != 3) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::ChannelsLast1d;
        orderOut = {0, 2, 1};
        reverseOrder = {0, 2, 1};
    } else if (src.isContiguous(diopiMemoryFormat_t::Contiguous) && destMemoryFormat == diopiMemoryFormat_t::ChannelsLast1d) {
        if (src.dim() != 3) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::Contiguous;
        orderOut = {0, 2, 1};
        reverseOrder = {0, 2, 1};
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast) && destMemoryFormat == diopiMemoryFormat_t::Contiguous) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::ChannelsLast;
        orderOut = {0, 3, 1, 2};
        reverseOrder = {0, 2, 3, 1};
    } else if (src.isContiguous(diopiMemoryFormat_t::Contiguous) && destMemoryFormat == diopiMemoryFormat_t::ChannelsLast) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::Contiguous;
        orderOut = {0, 2, 3, 1};
        reverseOrder = {0, 3, 1, 2};
    } else if (src.isContiguous(diopiMemoryFormat_t::Contiguous) && destMemoryFormat == diopiMemoryFormat_t::ChannelsLast3d) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::Contiguous;
        orderOut = {0, 2, 3, 4, 1};
        reverseOrder = {0, 4, 1, 2, 3};
    } else if (src.isContiguous(diopiMemoryFormat_t::ChannelsLast3d) && destMemoryFormat == diopiMemoryFormat_t::Contiguous) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = diopiMemoryFormat_t::ChannelsLast3d;
        orderOut = {0, 4, 1, 2, 3};
        reverseOrder = {0, 2, 3, 4, 1};
    } else {
        // convert to contiguous format
        srcMemoryFormatOut = diopiMemoryFormat_t::Preserve;
        return diopiSuccess;
    }
    return diopiSuccess;
}

diopiError_t calCnnlLayout(diopiMemoryFormat_t memoryFormat, int64_t dim, cnnlTensorLayout_t& cnnlLayout) {
    switch (memoryFormat) {
        case diopiMemoryFormat_t::ChannelsLast1d:
            cnnlLayout = CNNL_LAYOUT_NLC;
        case diopiMemoryFormat_t::ChannelsLast:
            cnnlLayout = CNNL_LAYOUT_NHWC;
            break;
        case diopiMemoryFormat_t::ChannelsLast3d:
            cnnlLayout = CNNL_LAYOUT_NDHWC;
            break;
        case diopiMemoryFormat_t::Contiguous:
            if (dim == 4) {
                cnnlLayout = CNNL_LAYOUT_NCHW;
            } else if (dim == 5) {
                cnnlLayout = CNNL_LAYOUT_NCDHW;
            } else {
                setLastErrorString("memoryFormat (%d) is not matched.", memoryFormat);
                return diopiNoImplement;
            }
            break;
        default:
            setLastErrorString("memoryFormat (%d) is not matched.", memoryFormat);
            return diopiNoImplement;
    }
    return diopiSuccess;
}

// static bool hasZero(std::vector<int64_t> vec) {
//     return std::any_of(vec.begin(), vec.end(), [](auto i) { return i == 0; });
// }

template <typename T>
static std::vector<T> changeVecAccordingToOrder(std::vector<T> vec, std::vector<int32_t> order) {
    DIOPI_CHECK_ABORT(order.size() == vec.size(), "order's len is not equal vec's len");
    std::vector<T> newVec(vec.size(), 0);
    int j = 0;
    for (auto i : order) {
        newVec[j++] = vec[i];
    }
    return newVec;
}

std::vector<int64_t> calContiguousStride(std::vector<int64_t> shape) {
    int32_t len = shape.size();
    std::vector<int64_t> stride(len, 1);
    int64_t strideTmp = 1;
    for (int i = 0; i < len; ++i) {
        if (i > 0) {
            strideTmp *= shape[i - 1];
        }
        stride[len - i - 1] = strideTmp;
    }
    return stride;
}

// change the shape and stride, the stride is incremental.
// order: 0, 2, 3, 1
// shape: 2,3,4,5 stride: 60, 1, 15, 3  -->
// shape: 2,4,5,3 stride: 60, 15, 3, 1
diopiError_t permuteTensor(DiopiTensor& t, const std::vector<int32_t>& order) {
    // only change the shape but not change the stride.
    std::vector<int64_t> newShape = changeVecAccordingToOrder(t.shape(), order);
    std::vector<int64_t> newStride = calContiguousStride(t.shape());
    t.asStrided(newShape, newStride);
    return diopiSuccess;
}

/* Inplace contiguous, support NCHW <-> NHWC, NCDHW <-> NDHWC */
diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, diopiMemoryFormat_t memoryFormat) {
    if (src.isContiguous(memoryFormat)) {
        return diopiSuccess;
    }
    int64_t dim = src.dim();
    DIOPI_CHECK(dim <= 8, "only support less than 8d tensor currently");
    diopiMemoryFormat_t srcMemoryFormat;
    std::vector<int32_t> order;
    std::vector<int32_t> reverseOrder;
    DiopiTensor dest;
    DIOPI_CALL(calOrderAndSrcMemoryFormat(src, memoryFormat, srcMemoryFormat, order, reverseOrder));
    if (srcMemoryFormat == diopiMemoryFormat_t::Preserve) {
        DIOPI_CALL(clone(ctx, src, dest, memoryFormat));
        src = dest;
        return diopiSuccess;
    }
    dest = requiresTensor(ctx, src.shape(), src.dtype(), memoryFormat);
    // set CNNL_LAYOUT_ARRAY because NLC->NCL failed ( no layout NCL);
    cnnlTensorLayout_t srcLayout = CNNL_LAYOUT_ARRAY;
    cnnlTensorLayout_t destLayout = CNNL_LAYOUT_ARRAY;

    std::vector<int64_t> olderDestStride = dest.stride();
    std::vector<int64_t> olderDestShape = dest.shape();
    if (memoryFormat != diopiMemoryFormat_t::Contiguous) {
        DIOPI_CALL(permuteTensor(dest, order));
    } else {
        DIOPI_CALL(permuteTensor(src, reverseOrder));
    }
    DIOPI_CALL(transpose(ctx, src, dest, srcLayout, destLayout, order));
    // recovery the shape
    dest.asStrided(olderDestShape, olderDestStride);
    src = dest;
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
