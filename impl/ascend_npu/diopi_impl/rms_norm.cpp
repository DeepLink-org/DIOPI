/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#inclue "cmath"

namespace OP_IMPL_NS {
DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
                                    diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps) {
    BEGIN_CALL_ACL_OP(input, weight, bias, out, invRms);
    acl_op::norm_out(inputAt, 2, -1, true, invRmsAt);

    int normShapeSize = 1;
    for (int i = 0; i < normalizedShape.len; i++) {
        normShapeSize *= normalizedShape.data[i];
    }
    acl_op::mul_(invRmsAt, 1.0 / sqrt(normShapeSize));
    acl_op::add_(invRmsAt, eps);
    acl_op::reciprocal_(invRmsAt);

    acl_op::mul_out(inputAt, invRmsAt, outAt);
    acl_op::addcmul_out(biasAt, outAt, weightAt, out);
    END_CALL_ACL_OP();
}
DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                            diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t inv_rms,
                                            diopiSize_t normalized_shape, double eps) {
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
