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
diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                        diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay,
                        int64_t step, bool amsgrad) {
    std::cout << "calling ascend diopiAdamw" << std::endl;
    AscendTensor inputAt(input);
    if (!inputAt.defined() || inputAt.nueml() == 0) {
        return diopiSuccess;
    }
    AscendTensor gradAt(grad);
    AscendTensor expAvgAt(exp_avg);
    AscendTensor expAvgSqAt(exp_avg_sq);
    AscendTensor maxExpAvgSqAt(max_exp_avg_sq);
    diopiScalar_t beta1Scalar = constructDiopiScalarT(inputAt.dtype(), beta1);
    diopiScalar_t beta2Scalar = constructDiopiScalarT(inputAt.dtype(), beta2);
    diopiScalar_t beta1PowerScalar = constructDiopiScalarT(inputAt.dtype(), pow(beta1, step));
    diopiScalar_t beta2PowerScalar = constructDiopiScalarT(inputAt.dtype(), pow(beta2, step));
    diopiScalar_t lrScalar = constructDiopiScalarT(inputAt.dtype(), lr);
    diopiScalar_t weightDecayScalar = constructDiopiScalarT(inputAt.dtype(), weight_decay);
    diopiScalar_t epsScalar = constructDiopiScalarT(inputAt.dtype(), eps);

    AclOpRunner<12, 3>("ApplyAdamW", ctx) adamwRunner.addInput(inputAt)
        .addInput(expAvgAt)
        .addInput(expAvgSqAt)
        .addInput(beta1PowerScalar)
        .addInput(beta2PowerScalar)
        .addInput(lrScalar)
        .addInput(weightDecayScalar)
        .addInput(beta1Scalar)
        .addInput(beta2Scalar)
        .addInput(epsScalar)
        .addInput(gradAt);

    if (maxExpAvgSq.defined()) {
        adamwRunner.addInput(maxExpAvgSqAt);
    }

    adamwRunner.setAttr("amsgrad", amsgrad).setAttr("maximize", false).addOutput(inputAt).addOutput(expAvgAt).addOutput(expAvgSqAt);
    adamwRunner.run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
