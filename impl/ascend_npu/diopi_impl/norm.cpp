/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

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

}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    BEGIN_CALL_ACL_OP(input, out, p, dim);
    if (!outAt.defined() || outAt.numel() <= 0) {
        return diopiSuccess;
    }

    if (!inputAt.defined()) {
        acl_op::fill_(outAt, 0);
        return diopiSuccess;
    }
    TORCH_CHECK(inputAt.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(outAt.scalar_type() == at::ScalarType::Float);
    bool keepdim = outAt.dim() == inputAt.dim();
    auto pvalue = calculateP(pAt);
    at_npu::native::OpCommand cmd1;
    cmd1.Name("LpNormReduceV2")
        .Input(inputAt)
        .Output(outAt)
        .Attr("p", pvalue)
        .Attr("axes", dimAt)
        .Attr("keepdim", keepdim)
        .Attr("epsilon", static_cast<float>(0))
        .Run();

    at_npu::native::OpCommand cmd2;
    cmd2.Name("LpNormUpdateV2").Input(outAt).Output(outAt).Attr("p", pvalue).Attr("epsilon", static_cast<float>(0)).Run();
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
