/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t numGroups,
                                      double eps) {
    if (0 == AscendTensor(input).numel()) {
        AclOpRunner<1, 1>("Fills", ctx).addInput(out).setAttr<float>("value", 0).addOutput(out).run();
        return diopiSuccess;
    }

    AclOpRunner<3, 3>("GroupNorm", ctx)
        .addInput(input)
        .addInput(weight)
        .addInput(bias)
        .setAttr("num_groups", static_cast<int32_t>(numGroups))
        .setAttr("epsilon", static_cast<float>(eps))
        .setAttr("data_format", std::string{getAclDataFormat(input) > 2 ? "NCHW" : "ND"})
        .setAttr("is_training", true)
        .addOutput(out)
        .addOutput(saveMean)
        .addOutput(saveInvstd)
        .run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
