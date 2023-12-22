/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
void removeTheFrontOneInShape(DiopiTensor& src, const std::vector<int64_t>& destShape) {
    int64_t srcDim = src.dim();
    int64_t destDim = destShape.size();
    if (srcDim <= destDim) {
        return;
    }
    // remove the front one in shape
    int64_t diffDim = srcDim - destDim;
    bool canRemoveFlag = true;
    for (int i = 0; i < diffDim; ++i) {
        if (src.shape()[i] != 1) {
            canRemoveFlag = false;
            break;
        }
    }
    int64_t offset = canRemoveFlag ? diffDim : 0;
    std::vector<int64_t> newSrcShape(src.shape().begin() + offset, src.shape().end());
    std::vector<int64_t> newSrcStrides(src.stride().begin() + offset, src.stride().end());
    src.asStrided(newSrcShape, newSrcStrides);
    return;
}

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    if (src == dest) {
        // the same address of pointers, return earlier
        return diopiSuccess;
    }
    DiopiTensor srcTr(src);
    DiopiTensor destTr(dest);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    if (!srcTr.defined()) {
        return diopiSuccess;
    }
    DIOPI_CHECK(destTr.defined(), "dest is not defined but src is defined.")
    if (destTr.numel() == 0) {
        return diopiSuccess;
    }

    // memory format convert if memory format is matched.
    // cnnTranspose doesn't support float64 and scalar and contiguousOut only support convertion between the contiguous tensor and the no-contiguous tensor.
    if (srcTr.shape() == destTr.shape() && srcTr.dim() != 0 && srcTr.dtype() != diopi_dtype_float64 && denseCheck(srcTr) && denseCheck(destTr) &&
        (destTr.isContiguous() || srcTr.isContiguous())) {
        DiopiTensor destTmpTr = destTr;
        if (destTmpTr.dtype() != srcTr.dtype()) {
            destTmpTr = requiresTensor(ctx, destTr.shape(), srcTr.dtype());
        }
        DIOPI_CALL(contiguousOut(ctx, srcTr, destTmpTr));
        if (destTmpTr.dtype() != destTr.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, destTr, destTmpTr));
        }
        return diopiSuccess;
    }

    // Ordinary copy
    // broadcast
    if (srcTr.shape() != destTr.shape()) {
        std::vector<int64_t> destStrides;
        DiopiTensor srcBroadcasted;
        removeTheFrontOneInShape(srcTr, destTr.shape());  // remove this when some ops (max_pool2d etc.) are refactored by right shape.
        if (broadcast(srcTr, destTr.shape(), &srcBroadcasted)) {
            srcTr = srcBroadcasted;
        } else {
            DIOPI_CHECK(false,
                        "can't broadcast because of the mismatched shape, src's shape: (%s), the dest's shape: (%s)",
                        vec2str(srcTr.shape()).c_str(),
                        vec2str(destTr.shape()).c_str());  // return
            return diopiErrorOccurred;
        }
    }

    // data type cast
    if (srcTr.dtype() != destTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, srcTr, destTr.dtype()));
    }
    CnnlTensorDesc inputDesc(destTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlCopy(handle, srcDesc.get(), srcTr.data(), inputDesc.get(), destTr.data()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
