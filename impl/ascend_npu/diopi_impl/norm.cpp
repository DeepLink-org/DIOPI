/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {

float calculateP(c10::optional<at::Scalar> p) {
    if (p.has_value()) {
        float val = at_npu::native::CalcuOpUtil::GetScalarFloatValue(p.value());
        if (val == INFINITY) {
            return static_cast<float>(INT_MAX);  // p = inf
        } else if (val == -INFINITY) {
            return static_cast<float>(INT_MIN);  // p = -inf
        } else {
            return p.value().toFloat();
        }
    } else {
        return static_cast<float>(2.0);  // default: p = 2.0
    }
}

inline bool checkUseAclop(float pfloat) {
    if (pfloat != 0.0 && pfloat != 1.0 && pfloat != 2.0 && pfloat != 3.0) {
        return true;
    }
    return false;
}

}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    BEGIN_CALL_ACL_OP(input, out, p, dim);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (!inputAt.defined()) {
        op_api::fill_(outAt, 0);
        return diopiSuccess;
    }

    bool keepdim = outAt.dim() == inputAt.dim();
    auto pvalue = calculateP(pAt);
    if (pvalue == 0) {
        op_api::fill_(outAt, inputAt.numel());
        return diopiSuccess;
    }
    bool useAclop = checkUseAclop(pvalue);
    if (useAclop) {
        op_api::norm_out(inputAt, pAt, dimAt, keepdim, outAt);
    } else {
        EXEC_NPU_CMD(aclnnNorm, inputAt, pAt, dimAt, keepdim, outAt);
    }

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
