/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

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
    eps = (eps < 1e-5) ? 1e-5 : eps;

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
        diopiScalar_t zeroScalar = constructDiopiScalarT(diopi_dtype_float64, 0.0);
        diopiFill(ctx, gradBias, &zeroScalar);
        if (inputAt.shape()[0] == 0) {
            diopiFill(ctx, gradWeight, &zeroScalar);
        } else {
            fillNan(ctx, gradWeightAt);
        }
    } else {
        int64_t n = inputAt.shape(0);
        int64_t c = inputAt.shape(1);
        int64_t hw = inputAt.numel() / (n * c);
        int64_t gradMaskData[3] = {true, true, true};
        if (nullptr == gradInput) {
            gradMaskData[0] = false;
        }
        if (nullptr == gradWeight) {
            gradMaskData[1] = false;
        }
        if (nullptr == gradBias) {
            gradMaskData[2] = false;
        }
        diopiSize_t gradMask{gradMaskData, 3};
        DIOPI_ASCEND_CALL_ACLNN(
            aclnnGroupNormBackward, ctx, gradOutput, inputAt, mean, rstd, weight, n, c, hw, numGroups, gradMask, gradInput, gradWeightAt, gradBias);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
