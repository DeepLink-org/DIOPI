/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    BEGIN_CALL_ACL_OP(input, mat2, out);

    if (inputAt.numel() == 0 || mat2At.numel() == 0) {
        op_api::fill_(outAt, c10::Scalar(0.0));
        END_CALL_ACL_OP();
    }

    op_api::mm_out(inputAt, mat2At, outAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
