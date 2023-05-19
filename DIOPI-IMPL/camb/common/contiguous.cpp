/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t transpose(diopiContextHandle_t& ctx, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut,
                       std::vector<int> order) {
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

void generateLayoutOrder(int64_t dim, MemoryFormat memoryFormat, cnnlTensorLayout_t& layoutIn, cnnlTensorLayout_t& layoutOut, std::vector<int>& order) {
    if (memoryFormat == MemoryFormat::Contiguous) {
        if (dim == 4) {
            layoutIn = CNNL_LAYOUT_NHWC;
            layoutOut = CNNL_LAYOUT_NCHW;
            order = {0, 3, 1, 2};
        } else if (dim == 5) {
            layoutIn = CNNL_LAYOUT_NDHWC;
            layoutOut = CNNL_LAYOUT_NCDHW;
            order = {0, 4, 1, 2, 3};
        }
    } else if (memoryFormat == MemoryFormat::ChannelsLast) {
        if (dim == 4) {
            layoutIn = CNNL_LAYOUT_NCHW;
            layoutOut = CNNL_LAYOUT_NHWC;
            order = {0, 2, 3, 1};
        }
    } else if (memoryFormat == MemoryFormat::ChannelsLast3d) {
        if (dim == 5) {
            layoutIn = CNNL_LAYOUT_NCDHW;
            layoutOut = CNNL_LAYOUT_NDHWC;
            order = {0, 2, 3, 4, 1};
        }
    }
}

/* Inplace contiguous, support NCHW <-> NHWC, NCDHW <-> NDHWC */
diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memoryFormat) {
    if (src.is_contiguous(memoryFormat)) return diopiSuccess;

    int64_t dim = src.dim();
    DIOPI_CHECK(dim == 4 || dim == 5, "only support 4d/5d tensor currently");

    cnnlTensorLayout_t layoutIn, layoutOut;
    std::vector<int> order;

    generateLayoutOrder(dim, memoryFormat, layoutIn, layoutOut, order);

    DiopiTensor dest = requiresTensor(ctx, src.shape(), src.dtype(), memoryFormat);
    DIOPI_CALL(transpose(ctx, src, dest, layoutIn, layoutOut, order));
    // DIOPI_CALL(diopiCopyInp(ctx, src.tensorHandle(), dest.tensorHandle()));
    src = dest;
    return diopiSuccess;
}

/* Inplace contiguous, support special layout like CNNL_LAYOUT_HWCN */
diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memoryFormat, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut) {
    if (src.is_contiguous(memoryFormat)) return diopiSuccess;
    DIOPI_CHECK(src.dim() == 4, "only support 4d tensor currently");

    std::vector<int> order;
    if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        DIOPI_CHECK(false,
                    "unkown layout error, layout should be "
                    "in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN]");
    }

    DiopiTensor dest = requiresTensor(ctx, src.shape(), src.dtype(), memoryFormat);
    DIOPI_CALL(transpose(ctx, src, dest, layoutIn, layoutOut, order));
    src = dest;
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
