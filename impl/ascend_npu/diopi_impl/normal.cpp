/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator);
    if (out == nullptr || outAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::normal_out(mean, std, outAt.sizes(), generatorAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(inout, generator);
    if (inout == nullptr || inoutAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::normal_(inoutAt, mean, std, generatorAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                     diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, std);
    if (out == nullptr || outAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::normal_out(mean, stdAt, generatorAt, outAt);
    END_CALL_ACL_OP();
}

// aclnn impl can't pass the ks test, failed to execute normal
diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                               diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, mean, std);
    if (out == nullptr || outAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::normal_out(meanAt, stdAt, generatorAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                     diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator, mean);
    if (out == nullptr || outAt.numel() == 0) {
        return diopiSuccess;
    }
    op_api::normal_out(meanAt, std, generatorAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
