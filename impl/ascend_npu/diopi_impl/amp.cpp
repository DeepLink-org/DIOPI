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
    op_api::_amp_foreach_non_finite_check_and_unscale_(scaledGradsList, foundInfAt, invScaleAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
