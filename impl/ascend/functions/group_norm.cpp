/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t saveMean, diopiTensorHandle_t saveInvstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t numGroups,
                                      double eps) {
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    int64_t n = inputAt.shape(0);
    int64_t c = inputAt.shape(1);
    int64_t hw = inputAt.numel() / (n * c);

    DIOPI_ASCEND_CALL_ACLNN(aclnnGroupNorm, ctx, input, weight, bias, n, c, hw, numGroups, eps, out, saveMean, saveInvstd);
    return diopiSuccess;
}

diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                    diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t numGroups) {
    AscendTensor inputAt(input);
    AscendTensor gradWeightAt(gradWeight);

    if (!inputAt.defined()) {
        return diopiSuccess;
    }

    if (inputAt.numel() == 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradBias);
        if (inputAt.shape(0) == 0 || inputAt.shape(1) == 0) {
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradWeight);
        } else {
            diopiScalar_t nanScalar = constructDiopiScalarT(diopi_dtype_float32, std::nanf(""));
            DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceFillScalar, ctx, gradWeightAt, &nanScalar);
        }
    } else {
        int64_t n = inputAt.shape(0);
        int64_t c = inputAt.shape(1);
        int64_t hw = inputAt.numel() / (n * c);

        std::array<bool, 3> gradMask = {gradInput != nullptr, gradWeight != nullptr, gradBias != nullptr};
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnGroupNormBackward, ctx, gradOutput, inputAt, mean, rstd, weight, n, c, hw, numGroups, gradMask, gradInput, gradWeightAt, gradBias);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
