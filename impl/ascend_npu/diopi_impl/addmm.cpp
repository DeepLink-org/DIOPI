/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    std::cout << "hellllllllllllllllllllll0" << std::endl;
    BEGIN_CALL_ACL_OP(out, input, mat1, mat2, beta, alpha);
    at_npu::native::OpPreparation::markAsOutputForApplyTensor(outAt);
    outAt = acl_op::addmm(inputAt, mat1At, mat2At, betaAt, alphaAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
