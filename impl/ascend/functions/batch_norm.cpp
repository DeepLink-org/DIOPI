#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

namespace {
diopiError_t batch_norm_training_reduce(diopiContextHandle_t ctx, diopiTensorHandle_t sum, diopiTensorHandle_t square_sum, diopiConstTensorHandle_t input,
                                        double eps) {
    AclOpRunner<1, 2>("BNTrainingReduce").addInput(input).setAttr("epsilon", static_cast<float>(eps)).addOutput(sum, square_sum).run(ctx);
    return diopiSuccess;
}

diopiError_t batch_norm_training_update(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                        diopiConstTensorHandle_t input, diopiTensorHandle_t sum, diopiTensorHandle_t square_sum,
                                        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                        diopiTensorHandle_t running_var, double momentum, double eps) {
    AclOpRunner<7, 5> runner("BNTrainingUpdate");
    runner.addInput(input, sum, square_sum, weight, bias, running_mean, running_var);
    runner.setAttr("epsilon", static_cast<float>(eps)).setAttr("factor", static_cast<float>(momentum));
    runner.addOutput(out, running_mean, running_mean, save_mean, save_invstd);
    runner.run(ctx);
    return diopiSuccess;
}

diopiError_t batch_norm_infer(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                              diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, double eps) {
    AclOpRunner<5, 1> runner("BNInfer");
    runner.addInput(input, weight, bias, running_mean, running_var);
    runner.setAttr("epsilon", static_cast<float>(eps));
    runner.addOutput(out);
    runner.run(ctx);
    return diopiSuccess;
}
// Backward

diopiError_t batch_norm_backward_training_reduce(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                 diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                 diopiConstTensorHandle_t weight, diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd,
                                                 double eps) {
    AclOpRunner<7, 1> runner("BNTrainingReduceGrad");
    runner.addInput(grad_output, input, grad_weight, grad_bias, weight, save_mean, save_invstd);
    runner.setAttr("epsilon", static_cast<float>(eps));
    runner.addOutput(grad_input);
    runner.run(ctx);
    return diopiSuccess;
}

diopiError_t batch_norm_backward_training_update(diopiContextHandle_t ctx, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                                 diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t save_mean,
                                                 diopiConstTensorHandle_t save_invstd, double eps) {
    AclOpRunner<4, 2> runner("BNTrainingUpdateGrad");
    runner.addInput(grad_output, input, save_mean, save_invstd);
    runner.setAttr("epsilon", static_cast<float>(eps));
    runner.addOutput(grad_weight, grad_bias);
    runner.run(ctx);
    return diopiSuccess;
}

diopiError_t batch_norm_infer(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t weight,
                              diopiConstTensorHandle_t running_var, double eps) {
    AclOpRunner<3, 1> runner("BNInferGrad");
    runner.addInput(grad_output, weight, running_var);
    runner.setAttr("epsilon", static_cast<float>(eps));
    runner.addOutput(grad_input);
    runner.run(ctx);
    return diopiSuccess;
}

}  // namespace

/**
 * @brief Applies Batch Normalization for each channel across a batch of data.
 * @param[in] ctx Context environment.
 * @param input input tensor. type = [float32, float16, float64].
 * @param weight weight tensor. type = [float32, float16, float64].
 * @param bias bias tensor. type = [float32, float16, float64].
 * @param running_mean weighted average tensor. type = [float32, float16, float64].
 * @param running_var weighted variance tensor. type = [float32, float16, float64].
 * @param training check if in training mode.
 * @param momentum Used to calculate the running mean and variance during runtime. type = [float32, float64]
 * @param eps The value added to the denominator during batch normalization to ensure numerical stability. type = [float32, float64]
 * @param[out] out normalized result. type = [float32, float16, float64].
 * @param save_mean Mean tensor,the mean value for each feature channel of the input tensor. type = [float32, float16, float64].
 * @param save_invstd Backup of inverse standard deviation computed during training. type = [float32, float16, float64].
 */
extern "C" DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                                 diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                                 diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var,
                                                 bool training, double momentum, double eps) {
    if (!training) {
        CALL_ACLRT(batch_norm_infer(ctx, out, input, weight, bias, running_mean, running_var, eps));
        return diopiSuccess;
    }

    diopiTensorHandle_t sum = nullptr, square_sum = nullptr;
    makeTensorFromTensor(ctx, &sum, running_mean);
    makeTensorFromTensor(ctx, &square_sum, running_mean);

    CALL_ACLRT(batch_norm_training_reduce(ctx, sum, square_sum, input, eps));

    CALL_ACLRT(batch_norm_training_update(ctx, out, save_mean, save_invstd, input, sum, square_sum, weight, bias, running_mean, running_var, momentum, eps));

    return diopiSuccess;
}

/**
 * @brief compute the backward pass of batch normalization
 * @param[in] grad_output Gradient of normalized layer output, with the same shape as the forward pass output. type=[float32, float16, float64].
 * @param[out] grad_input Gradient of the input data, with the same shape as the input data. type = [float32, float16, float64].
 * @param grad_weight Gradient of the weight parameter, with the same shape as the weight parameter. type = [float32, float16, float64].
 * @param grad_bias Gradient of the bias parameter, with the same shape as the bias parameter. type = [float32, float16, float64].
 * @sa Other parameters refer to diopiBatchNorm().
 */
extern "C" DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                         diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                         diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean,
                                                         diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                                         diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    if (!training) {
        CALL_ACLRT(batch_norm_infer(ctx, grad_input, grad_output, weight, running_var, eps));
        return diopiSuccess;
    }
    
    CALL_ACLRT(batch_norm_backward_training_update(ctx, grad_weight, grad_bias, grad_output, input, save_mean, save_invstd, eps));

    CALL_ACLRT(batch_norm_backward_training_reduce(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, save_mean, save_invstd, eps));

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
