/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    AscendTensor inputAt(input);
    AscendTensor indexAt(index);
    AscendTensor outAt(out);

    if (inputAt.numel() == 0 || indexAt.numel() == 0) {
        return diopiSuccess;
    }

    if (inputAt.dim() == 0) {
        inputAt.unsqueeze(0);
    }

    if (indexAt.dim() == 0) {
        indexAt.unsqueeze(0);
    }

    if (outAt.dim() == 0) {
        outAt.unsqueeze(0);
    }

    // alcnnGather requires that the dimensions of input and index match.
    ASCEND_CHECK_ABORT(inputAt.dim() == indexAt.dim(), "alcnnGather requires that the dimensions of input and index match.");
    // aclnnGather requires that the shape of index and out must be the same.
    ASCEND_CHECK_ABORT(indexAt.shape() == outAt.shape(), "aclnnGather requires that the shape of index and out must be the same.");


    DIOPI_ASCEND_CALL_ACLNN(aclnnGather, ctx, inputAt, dim, indexAt, outAt);
    return diopiSuccess;
}

diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                 int64_t dim, diopiConstTensorHandle_t index) {
    AscendTensor inputAt(input);
    AscendTensor gradInputAt(gradInput);
    AscendTensor gradOutputAt(gradOutput);
    AscendTensor indexAt(index);

    if (inputAt.numel() == 0 || indexAt.numel() == 0) {
        return diopiSuccess;
    }

    if (indexAt.dim() == 0) {
        indexAt.unsqueeze(0);
    }

    if (gradInputAt.dim() == 0) {
        gradInputAt.unsqueeze(0);
    }

    if (gradOutputAt.dim() == 0) {
        gradOutputAt.unsqueeze(0);
    }

    // alcnnScatter requires that the dimensions of input and index match.
    ASCEND_CHECK_ABORT(gradInputAt.dim() == indexAt.dim(), 
        "alcnnScatter requires that the dimensions of input and index match.");
    // aclnnScatter requires that the shape of index and out must be the same.
    ASCEND_CHECK_ABORT(indexAt.shape() == gradOutputAt.shape(), 
        "aclnnScatter requires that the shape of index and out must be the same.");

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInputAt);
    // the gradOutput will accumulate in gradInput according to index.
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterAdd, ctx, gradInputAt, dim, indexAt, gradOutputAt, gradInputAt);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
