/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <math.h>

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t expAvg, diopiTensorHandle_t expAvgSq,
                        diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay, int64_t step, bool amsgrad) {
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }
    AscendTensor gradAt(grad);
    AscendTensor expAvgAt(expAvg);
    AscendTensor expAvgSqAt(expAvgSq);
    AscendTensor maxExpAvgSqAt(maxExpAvgSq);
    diopiScalar_t beta1Scalar = constructDiopiScalarT(inputAt.dtype(), beta1);
    diopiScalar_t beta2Scalar = constructDiopiScalarT(inputAt.dtype(), beta2);
    diopiScalar_t beta1PowerScalar = constructDiopiScalarT(inputAt.dtype(), pow(beta1, step));
    diopiScalar_t beta2PowerScalar = constructDiopiScalarT(inputAt.dtype(), pow(beta2, step));
    diopiScalar_t lrScalar = constructDiopiScalarT(inputAt.dtype(), lr);
    diopiScalar_t weightDecayScalar = constructDiopiScalarT(inputAt.dtype(), weightDecay);
    diopiScalar_t epsScalar = constructDiopiScalarT(inputAt.dtype(), eps);
    diopiDtype_t inputDtype = inputAt.dtype();

    AclOpRunner<12, 3> adamwRunner("ApplyAdamW", ctx);
    adamwRunner.addInput(inputAt)
        .addInput(expAvgAt)
        .addInput(expAvgSqAt)
        .addConstInput(beta1PowerScalar, inputDtype)
        .addConstInput(beta2PowerScalar, inputDtype)
        .addConstInput(lrScalar, inputDtype)
        .addConstInput(weightDecayScalar, inputDtype)
        .addConstInput(beta1Scalar, inputDtype)
        .addConstInput(beta2Scalar, inputDtype)
        .addConstInput(epsScalar, inputDtype)
        .addInput(gradAt);

    if (maxExpAvgSqAt.defined()) {
        adamwRunner.addInput(maxExpAvgSqAt);
    }

    adamwRunner.setAttr("amsgrad", amsgrad).setAttr("maximize", false).addOutput(inputAt).addOutput(expAvgAt).addOutput(expAvgSqAt);
    adamwRunner.run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
