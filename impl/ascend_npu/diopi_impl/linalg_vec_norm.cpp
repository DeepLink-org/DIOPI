/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiLinalgVecNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiScalar_t* ord, diopiSize_t dim,
                                bool keepdim) {
    BEGIN_CALL_ACL_OP(input, out, ord, dim);
    if (inputAt.numel() <= 0) {
        op_api::fill_(outAt, at::Scalar(0));
        END_CALL_ACL_OP();
    }
    op_api::linalg_vector_norm_out(inputAt, ordAt, dimAt, keepdim, outAt.scalar_type(), outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
