/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    AscendTensor tensor(tensors[0]);
    if (!tensor.defined() || tensor.numel() == 0) {
        return diopiSuccess;
    }

    std::vector<diopiConstTensorHandle_t> tensorsVec(numInputs);
    for (int i = 0; i < numInputs; i++) {
        tensorsVec[i] = tensors[i];
    }
    DIOPI_ASCEND_CALL_ACLNN(aclnnStack, ctx, tensorsVec, dim, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
