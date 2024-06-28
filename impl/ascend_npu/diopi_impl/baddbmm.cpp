/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                          diopiConstTensorHandle_t batch2, double beta, double alpha) {
    BEGIN_CALL_ACL_OP(out, input, batch1, batch2);
    auto betaAt = at::Scalar(beta);
    auto alphaAt = at::Scalar(alpha);

    if (0 == outAt.numel()) {
        END_CALL_ACL_OP();
    }

    op_api::baddbmm_out(inputAt, batch1At, batch2At, betaAt, alphaAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta,
                             double alpha) {
    BEGIN_CALL_ACL_OP(input, batch1, batch2);
    auto betaAt = at::Scalar(beta);
    auto alphaAt = at::Scalar(alpha);

    if (0 == inputAt.numel()) {
        END_CALL_ACL_OP();
    }

    op_api::baddbmm_(inputAt, batch1At, batch2At, betaAt, alphaAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
