#include <diopi/functions.h>
#include <torch_br/csrc/diopi/DIOPIFrontend.h>

extern "C" {

DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                      diopiTensorHandle_t running_var, bool training, double momentum, double eps)
{
    brDiopiBatchNorm(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                              diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                              diopiConstTensorHandle_t save_invstd, bool training, double eps)
{
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value)
{
    return diopiSuccess;
}

}  // extern "C"