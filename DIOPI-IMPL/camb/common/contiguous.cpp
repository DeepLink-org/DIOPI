/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t transpose(diopiContextHandle_t& ctx, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layout_in, cnnlTensorLayout_t layout_out,
                       std::vector<int> order) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inDesc(in, layout_in);
    CnnlTensorDesc outDesc(out, layout_out);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspace_size));

    void* workspace_ptr = workspace_size == 0 ? requiresBuffer(ctx, workspace_size).data() : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(), in.data(), outDesc.get(), out.data(), workspace_ptr, workspace_size));
    return diopiSuccess;
}

void generate_layout_order(int64_t dim, MemoryFormat memory_format, cnnlTensorLayout_t& layout_in, cnnlTensorLayout_t& layout_out, std::vector<int>& order) {
    if (memory_format == MemoryFormat::Contiguous) {
        if (dim == 4) {
            layout_in = CNNL_LAYOUT_NHWC;
            layout_out = CNNL_LAYOUT_NCHW;
            order = {0, 3, 1, 2};
        } else if (dim == 5) {
            layout_in = CNNL_LAYOUT_NDHWC;
            layout_out = CNNL_LAYOUT_NCDHW;
            order = {0, 4, 1, 2, 3};
        }
    } else if (memory_format == MemoryFormat::ChannelsLast) {
        if (dim == 4) {
            layout_in = CNNL_LAYOUT_NCHW;
            layout_out = CNNL_LAYOUT_NHWC;
            order = {0, 2, 3, 1};
        }
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
        if (dim == 5) {
            layout_in = CNNL_LAYOUT_NCDHW;
            layout_out = CNNL_LAYOUT_NDHWC;
            order = {0, 2, 3, 4, 1};
        }
    }
}

/* Inplace contiguous, support NCHW <-> NHWC, NCDHW <-> NDHWC */
diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memory_format) {
    if (src.is_contiguous(memory_format)) return diopiSuccess;

    int64_t dim = src.dim();
    DIOPI_CHECK(dim == 4 || dim == 5, "only support 4d/5d tensor currently");

    cnnlTensorLayout_t layout_in, layout_out;
    std::vector<int> order;

    generate_layout_order(dim, memory_format, layout_in, layout_out, order);

    DiopiTensor dest = requiresTensor(ctx, src.shape(), src.dtype(), memory_format);
    DIOPI_CALL(transpose(ctx, src, dest, layout_in, layout_out, order));
    // DIOPI_CALL(diopiCopyInp(ctx, src.tensorHandle(), dest.tensorHandle()));
    src = dest;
    return diopiSuccess;
}

/* Inplace contiguous, support special layout like CNNL_LAYOUT_HWCN */
diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memory_format, cnnlTensorLayout_t layout_in, cnnlTensorLayout_t layout_out) {
    if (src.is_contiguous(memory_format)) return diopiSuccess;
    DIOPI_CHECK(src.dim() == 4, "only support 4d tensor currently");

    std::vector<int> order;
    if (layout_in == CNNL_LAYOUT_NHWC && layout_out == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layout_in == CNNL_LAYOUT_NCHW && layout_out == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layout_in == CNNL_LAYOUT_HWCN && layout_out == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layout_in == CNNL_LAYOUT_HWCN && layout_out == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        DIOPI_CHECK(false,
                    "unkown layout error, layout should be "
                    "in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN]");
    }

    DiopiTensor dest = requiresTensor(ctx, src.shape(), src.dtype(), memory_format);
    DIOPI_CALL(transpose(ctx, src, dest, layout_in, layout_out, order));
    src = dest;
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
