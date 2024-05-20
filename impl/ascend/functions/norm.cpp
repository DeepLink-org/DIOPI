/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    AscendTensor outTensor(out);
    if (!outTensor.defined() || outTensor.numel() == 0) {
        return diopiSuccess;
    }

    float pValue = getValue<float>(p);
    ASCEND_CHECK_ABORT(pValue == 0.0 || pValue == 1.0 || pValue == 2.0 || pValue == 3.0, "aclnnNorm currently only supports p=0,1,2,3!");
    bool keepDim = false;
    DIOPI_ASCEND_CALL_ACLNN(aclnnNorm, ctx, input, p, dim, keepDim, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
