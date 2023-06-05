/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <vector>

#include "common.hpp"
#include "debug.hpp"

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

static diopiError_t calTensorMemoryFormat(const DiopiTensor& tensor, MemoryFormat& memoryFormatOut) {
    if (tensor.isContiguous(MemoryFormat::ChannelsLast)) {
        memoryFormatOut = MemoryFormat::ChannelsLast;
    } else if (tensor.isContiguous(MemoryFormat::ChannelsLast3d)) {
        memoryFormatOut = MemoryFormat::ChannelsLast3d;
    } else if (tensor.isContiguous(MemoryFormat::Contiguous)) {
        memoryFormatOut = MemoryFormat::Contiguous;
    } else {
        return diopiNoImplement;
    }
    return diopiSuccess;
}

static diopiError_t calOrderAndSrcMemoryFormat(const DiopiTensor& src, MemoryFormat destMemoryFormat, MemoryFormat& srcMemoryFormatOut,
                                               std::vector<int32_t>& orderOut) {
    if (src.isContiguous(destMemoryFormat)) {
        srcMemoryFormatOut = destMemoryFormat;
        orderOut = {0, 1, 2, 3};
        return diopiSuccess;
    }
    if (src.isContiguous(MemoryFormat::ChannelsLast) && destMemoryFormat == MemoryFormat::Contiguous) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::ChannelsLast;
        orderOut = {0, 3, 1, 2};
    } else if (src.isContiguous(MemoryFormat::Contiguous) && destMemoryFormat == MemoryFormat::ChannelsLast) {
        if (src.dim() != 4) {
            setLastErrorString("the dim of the tensor should be 4, but now is %d", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::Contiguous;
        orderOut = {0, 2, 3, 1};
    } else if (src.isContiguous(MemoryFormat::Contiguous) && destMemoryFormat == MemoryFormat::ChannelsLast3d) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::Contiguous;
        orderOut = {0, 2, 3, 4, 1};
    } else if (src.isContiguous(MemoryFormat::ChannelsLast3d) && destMemoryFormat == MemoryFormat::Contiguous) {
        if (src.dim() != 5) {
            setLastErrorString("the dim of the tensor should be 5, but now is %d", src.dim());
            return diopiNoImplement;
        }
        srcMemoryFormatOut = MemoryFormat::ChannelsLast3d;
        orderOut = {0, 4, 1, 2, 3};
    } else {
        setLastErrorString("the memory format (%d) of tensor is not right!", static_cast<int32_t>(destMemoryFormat));
        return diopiNoImplement;
    }
    return diopiSuccess;
}

diopiError_t calCnnlLayout(MemoryFormat memoryFormat, int64_t dim, cnnlTensorLayout_t& cnnlLayout) {
    switch (memoryFormat) {
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

/* Inplace contiguous, support NCHW <-> NHWC, NCDHW <-> NDHWC */
diopiError_t contiguous(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memoryFormat) {
    if (src.isContiguous(memoryFormat)) {
        return diopiSuccess;
    };
    int64_t dim = src.dim();
    DIOPI_CHECK(dim == 4 || dim == 5, "only support 4d/5d tensor currently");

    DiopiTensor dest = requiresTensor(ctx, src.shape(), src.dtype(), memoryFormat);
    MemoryFormat srcMemoryFormat;
    std::vector<int32_t> order;
    DIOPI_CALL(calOrderAndSrcMemoryFormat(src, memoryFormat, srcMemoryFormat, order));
    cnnlTensorLayout_t srcLayout;
    cnnlTensorLayout_t destLayout;
    DIOPI_CALL(calCnnlLayout(srcMemoryFormat, dim, srcLayout));
    DIOPI_CALL(calCnnlLayout(memoryFormat, dim, destLayout));
    DIOPI_CALL(transpose(ctx, src, dest, srcLayout, destLayout, order));
    src = dest;
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
