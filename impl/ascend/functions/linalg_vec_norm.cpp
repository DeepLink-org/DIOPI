/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiLinalgVecNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiScalar_t* ord, diopiSize_t dim,
                                bool keepdim) {
    AscendTensor inputTensor(input);
    AscendTensor outTensor(out);
    if (inputTensor.numel() == 0) {
        diopiScalar_t value = constructDiopiScalarT(outTensor.dtype(), 0);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, out, &value);
    } else {
        DIOPI_ASCEND_CALL_ACLNN(aclnnLinalgVectorNorm, ctx, input, ord, dim, keepdim, outTensor.dtype(), out);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
