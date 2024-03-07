/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP(input, mat1, mat2, beta, alpha, out);
    op_api::addmm_out(inputAt, mat1At, mat2At, betaAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
