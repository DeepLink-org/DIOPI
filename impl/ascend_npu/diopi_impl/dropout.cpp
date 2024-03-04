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

    double keep_prob = 1 - p;

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

    auto results = op_api::_npu_dropout(inputAt, 1 - p);
    outAt.copy_(std::get<0>(results));

    auto outScaleAt = std::get<0>(results);
    auto scaleFactor = c10::Scalar(keep_prob);
    op_api::mul_(outScaleAt, scaleFactor);
    op_api::eq_out(inputAt, outScaleAt, maskAt);

    END_CALL_ACL_OP();
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(input, mask, generator);
    if (train) {
        diopiTensorHandle_t inputCopy;
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);
        diopiDtype_t dtype;
        diopiGetTensorDtype(input, &dtype);
        diopiDevice_t device;
        diopiGetTensorDevice(input, &device);
        diopiRequireTensor(ctx, &inputCopy, &inputShape, nullptr, dtype, device);
        diopiCopyInp(ctx, input, inputCopy);
        impl::ascend_npu::diopiDropout(ctx, input, mask, inputCopy, p, train, generator);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
