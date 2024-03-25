/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiAmpForeachNonFiniteCheckAndUnscaleInp(diopiContextHandle_t ctx, diopiTensorHandle_t* scaledGrads, int64_t numScaledGrads,
                                                        diopiTensorHandle_t foundInf, diopiConstTensorHandle_t invScale) {
    BEGIN_CALL_ACL_OP(foundInf, invScale);
    std::vector<at::Tensor> scaledGradsList;
    scaledGradsList.reserve(numScaledGrads);
    for (int i = 0; i < numScaledGrads; ++i) {
        scaledGradsList.emplace_back(impl::aten::buildATen(scaledGrads[i]));
    }

    DIOPI_CHECK(invScaleAt.numel() == 1, "inv_scale must be a 1-element tensor");
    DIOPI_CHECK(foundInfAt.numel() == 1, "found_inf must be a 1-element tensor");
    DIOPI_CHECK(invScaleAt.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor");
    DIOPI_CHECK(foundInfAt.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor");

    bool isFinite = true;
    for (const auto& scaledGrad : scaledGradsList) {
        if (!op_api::all(op_api::isfinite(scaledGrad)).item<bool>()) {
            isFinite = false;
            break;
        }
    }

    if (!isFinite) {
        op_api::ones_out(1, foundInfAt);
        return diopiSuccess;
    }

    for (auto& scaledGrad : scaledGradsList) {
        op_api::mul_(scaledGrad, invScaleAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
