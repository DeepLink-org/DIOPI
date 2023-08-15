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
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? requiresBuffer(ctx, workspaceSize).data() : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data(), workspacePtr, workspaceSize));
    return diopiSuccess;
}

// static diopiError_t calTensorMemoryFormat(const DiopiTensor& tensor, MemoryFormat& memoryFormatOut) {
//     if (tensor.isContiguous(MemoryFormat::ChannelsLast)) {
//         memoryFormatOut = MemoryFormat::ChannelsLast;
//     } else if (tensor.isContiguous(MemoryFormat::ChannelsLast3d)) {
//         memoryFormatOut = MemoryFormat::ChannelsLast3d;
//     } else if (tensor.isContiguous(MemoryFormat::Contiguous)) {
//         memoryFormatOut = MemoryFormat::Contiguous;
//     } else {
//         return diopiNoImplement;
//     }
//     return diopiSuccess;
// }

static diopiError_t calOrderAndSrcMemoryFormat(const DiopiTensor& src, MemoryFormat destMemoryFormat, MemoryFormat& srcMemoryFormatOut,
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
    if (src.isContiguous(MemoryFormat::ChannelsLast1d) && destMemoryFormat == MemoryFormat::Contiguous) {
        if (src.dim() != 3) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::ChannelsLast1d;
        orderOut = {0, 2, 1};
        reverseOrder = {0, 2, 1};
    } else if (src.isContiguous(MemoryFormat::Contiguous) && destMemoryFormat == MemoryFormat::ChannelsLast1d) {
        if (src.dim() != 3) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::Contiguous;
        orderOut = {0, 2, 1};
        reverseOrder = {0, 2, 1};
    } else if (src.isContiguous(MemoryFormat::ChannelsLast) && destMemoryFormat == MemoryFormat::Contiguous) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::ChannelsLast;
        orderOut = {0, 3, 1, 2};
        reverseOrder = {0, 2, 3, 1};
    } else if (src.isContiguous(MemoryFormat::Contiguous) && destMemoryFormat == MemoryFormat::ChannelsLast) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::Contiguous;
        orderOut = {0, 2, 3, 1};
        reverseOrder = {0, 3, 1, 2};
    } else if (src.isContiguous(MemoryFormat::Contiguous) && destMemoryFormat == MemoryFormat::ChannelsLast3d) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::Contiguous;
        orderOut = {0, 2, 3, 4, 1};
        reverseOrder = {0, 4, 1, 2, 3};
    } else if (src.isContiguous(MemoryFormat::ChannelsLast3d) && destMemoryFormat == MemoryFormat::Contiguous) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d.", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::ChannelsLast3d;
        orderOut = {0, 4, 1, 2, 3};
        reverseOrder = {0, 2, 3, 4, 1};
    } else {
        // convert to contiguous format
        srcMemoryFormatOut = MemoryFormat::Preserve;
        return diopiSuccess;
    }
    return diopiSuccess;
}

diopiError_t calCnnlLayout(MemoryFormat memoryFormat, int64_t dim, cnnlTensorLayout_t& cnnlLayout) {
    switch (memoryFormat) {
        case MemoryFormat::ChannelsLast1d:
            cnnlLayout = CNNL_LAYOUT_NLC;
        case MemoryFormat::ChannelsLast:
            cnnlLayout = CNNL_LAYOUT_NHWC;
            break;
        case MemoryFormat::ChannelsLast3d:
            cnnlLayout = CNNL_LAYOUT_NDHWC;
            break;
        case MemoryFormat::Contiguous:
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
diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, MemoryFormat memoryFormat) {
    if (src.isContiguous(memoryFormat)) {
        return diopiSuccess;
    }
    int64_t dim = src.dim();
    DIOPI_CHECK(dim <= 5, "only support less than 5d tensor currently");
    MemoryFormat srcMemoryFormat;
    std::vector<int32_t> order;
    std::vector<int32_t> reverseOrder;
    DiopiTensor dest;
    DIOPI_CALL(calOrderAndSrcMemoryFormat(src, memoryFormat, srcMemoryFormat, order, reverseOrder));
    if (srcMemoryFormat == MemoryFormat::Preserve) {
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
    std::vector<int64_t> olderSrcStride = src.stride();
    std::vector<int64_t> olderSrcShape = src.shape();
    if (memoryFormat != MemoryFormat::Contiguous) {
        DIOPI_CALL(permuteTensor(dest, order));
    } else {
        DIOPI_CALL(permuteTensor(src, reverseOrder));
    }
    DIOPI_CALL(transpose(ctx, src, dest, srcLayout, destLayout, order));
    // recovery the shape
    dest.asStrided(olderDestShape, olderDestStride);
    src.asStrided(olderSrcShape, olderSrcStride);
    src = dest;
    // printDevData(ctx, dest, "=====dest");
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
