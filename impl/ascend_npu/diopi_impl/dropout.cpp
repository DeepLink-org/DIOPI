/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train,
                          diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, input, mask, generator);

    if (p == 0 || train == false) {
        outAt.copy_(inputAt);
        op_api::fill_(maskAt, c10::Scalar(1));
        END_CALL_ACL_OP();
    }
    if (p == 1) {
        op_api::fill_(outAt, c10::Scalar(0));
        op_api::fill_(maskAt, c10::Scalar(0));
        END_CALL_ACL_OP();
    }

    if (inputAt.sizes() != maskAt.sizes()) {
        auto input2d = op_api::ones_like(maskAt, inputAt.scalar_type());
        auto results = op_api::_npu_dropout(input2d, p);
        op_api::mul_out(inputAt, std::get<0>(results), outAt);
        op_api::ne_out(std::get<0>(results), c10::Scalar(0), maskAt);
        END_CALL_ACL_OP();
    }

    auto results = op_api::_npu_dropout(inputAt, p);
    outAt.copy_(std::get<0>(results));
    op_api::ne_out(outAt, c10::Scalar(0), maskAt);

    END_CALL_ACL_OP();
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(input, mask, generator);

    if (p == 0 || train == false) {
        op_api::fill_(maskAt, c10::Scalar(1));
        END_CALL_ACL_OP();
    }
    if (p == 1) {
        op_api::fill_(inputAt, c10::Scalar(0));
        op_api::fill_(maskAt, c10::Scalar(0));
        END_CALL_ACL_OP();
    }
    if (inputAt.sizes() != maskAt.sizes()) {
        auto input2d = op_api::ones_like(maskAt, inputAt.scalar_type());
        auto results = op_api::_npu_dropout(input2d, p);
        op_api::mul_(inputAt, std::get<0>(results));
        op_api::ne_out(std::get<0>(results), c10::Scalar(0), maskAt);
        END_CALL_ACL_OP();
    }

    auto results = op_api::_npu_dropout(inputAt, p);
    inputAt.copy_(std::get<0>(results));
    op_api::ne_out(inputAt, c10::Scalar(0), maskAt);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
