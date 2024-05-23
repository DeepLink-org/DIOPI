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

    // alcnnGather requires that the dimensions of input, output, and index match.
    while (inputAt.dim() < indexAt.dim()) {
        inputAt.unsqueeze(0);
    }

    while (indexAt.dim() < inputAt.dim()) {
        indexAt.unsqueeze(0);
    }

    while (outAt.dim() < inputAt.dim()) {
        outAt.unsqueeze(0);
    }

    for(int64_t i = 0; i < inputAt.dim() ; i++) {
        if (inputAt.shape(i) == 0 || indexAt.shape(i) == 0) {
            return diopiSuccess;
        }
    }

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

    // alcnnScatter requires that the dimensions of input, output, and index match.
    while (indexAt.dim() < gradOutputAt.dim()) {
        indexAt.unsqueeze(0);
    }

    while (gradOutputAt.dim() < indexAt.dim()) {
        gradOutputAt.unsqueeze(0);
    }

    while (gradInputAt.dim() < gradOutputAt.dim()) {
        gradInputAt.unsqueeze(0);
    }
    
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInputAt);
    // the gradOutput will accumulate in gradInput according to index.
    DIOPI_ASCEND_CALL_ACLNN(aclnnAddScatter, ctx, gradInputAt, dim, indexAt, gradOutputAt, gradInput);                       
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
