/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                                 diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                                 diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var,
                                                 bool training, double momentum, double eps) {
    if (!training) {
        AclOpRunner<5, 1>("BNInfer")
            .addInput(input, weight, bias, running_mean, running_var)
            .addOutput(out)
            .setAttr("epsilon", static_cast<float>(eps))
            .run(ctx);
    } else {
        diopiTensorHandle_t sum = nullptr, square_sum = nullptr;
        diopiSize_t shape, stride;
        diopiGetTensorShape(running_mean, &shape);
        diopiGetTensorStride(running_mean, &stride);
        diopiRequireTensor(ctx, &sum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        diopiRequireTensor(ctx, &square_sum, &shape, &stride, diopiDtype_t::diopi_dtype_float32, diopi_device);
        AclOpRunner<1, 2>("BNTrainingReduce").addInput(input).setAttr("epsilon", static_cast<float>(eps)).addOutput(sum, square_sum).run(ctx);
        AclOpRunner<7, 5>("BNTrainingUpdate")
            .addInput(input, sum, square_sum, weight, bias, running_mean, running_var)
            .setAttr("epsilon", static_cast<float>(eps))
            .setAttr("factor", static_cast<float>(momentum))
            .addOutput(out, running_mean, running_mean, save_mean, save_invstd)
            .run(ctx);
    }
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                         diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                         diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean,
                                                         diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                                         diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    if (!training) {
        AclOpRunner<3, 1>("BNInferGrad").addInput(grad_output, weight, running_var).setAttr("epsilon", static_cast<float>(eps)).addOutput(grad_input).run(ctx);
    } else {
        AclOpRunner<4, 2>("BNTrainingUpdateGrad")
            .addInput(grad_output, input, save_mean, save_invstd)
            .setAttr("epsilon", static_cast<float>(eps))
            .addOutput(grad_weight, grad_bias)
            .run(ctx);
        AclOpRunner<7, 1>("BNTrainingReduceGrad")
            .addInput(grad_output, input, grad_weight, grad_bias, weight, save_mean, save_invstd)
            .setAttr("epsilon", static_cast<float>(eps))
            .addOutput(grad_input)
            .run(ctx);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
