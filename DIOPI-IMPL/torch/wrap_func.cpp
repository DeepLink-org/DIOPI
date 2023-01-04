#include <diopi/functions.h>
#include <stdio.h>
#include <dlfcn.h>

static void* handle;

static void
__attribute__ ((constructor))
diopi_init(void) {
    printf("diopi> init\n");
    handle = dlopen("libdiopi_real_impl.so", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!handle) {
        fprintf (stderr, "%s ", dlerror());
    }
}

static void
__attribute__ ((destructor))
diopi_fini(void)
{
  printf("diopi> fini\n");
  dlclose(handle);
}

DIOPI_RT_API const char* diopiGetVendorName() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetVendorName"));
    return (*func)();
}

DIOPI_RT_API const char* diopiGetImplVersion() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetImplVersion"));
    return (*func)();
}

DIOPI_RT_API const char* diopiGetLastErrorString() {
    const char* (*func)();
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGetLastErrorString"));
    return (*func)();
}

DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution2d"));
    return (*func)(ctx, out, input, weight, bias, stride, padding, dilation, groups);
}

DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t *, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution2dBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad3, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiConstTensorHandle_t running_mean,
                                      diopiConstTensorHandle_t running_var, bool training, double momentum, double eps) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, bool, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBatchNorm"));
    return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                              diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, 
                                              diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, bool, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBatchNormBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps);
}

DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRelu"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReluInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanh"));
    return (*func)(ctx, out, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanhInp"));
    return (*func)(ctx, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiHardtanhBackward"));
    return (*func)(ctx, grad_input, grad_output, input, min_val, max_val);
}

DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThreshold"));
    return (*func)(ctx, out, input, threshold, value);
}

DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThresholdInp"));
    return (*func)(ctx, input, threshold, value);
}

DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiThresholdBackward"));
    return (*func)(ctx, grad_input, grad_output, input, threshold);
}

DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, const char* approximate) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGelu"));
    return (*func)(ctx, out, input, approximate);
}

DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char* approximate) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeluBackward"));
    return (*func)(ctx, grad_input, grad_output, input, approximate);
}

DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyRelu"));
    return (*func)(ctx, out, input, negative_slope);
}

DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyReluInp"));
    return (*func)(ctx, input, negative_slope);
}

DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeakyReluBackward"));
    return (*func)(ctx, grad_input, grad_output, input, negative_slope, input_is_result);
}

DIOPI_API diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                      bool count_include_pad, const int64_t* divisor_override) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, bool,
        bool, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAvgPool2d"));
    return (*func)(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

DIOPI_API diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                              bool count_include_pad, const int64_t* divisor_override) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, bool,
        bool, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAvgPool2dBackward"));
    return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2d"));
    return (*func)(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
}

DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t,
        diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2dWithIndices"));
    return (*func)(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}

DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool2dBackward"));
    return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool2d"));
    return (*func)(ctx, out, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool2dBackward"));
    return (*func)(ctx, grad_input, grad_output, input);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2d"));
    return (*func)(ctx, out, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2dWithIndices"));
    return (*func)(ctx, out, indices, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool2dBackward"));
    return (*func)(ctx, grad_input, grad_output, input, indices);
}

DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask,
                                    diopiConstTensorHandle_t input, double p, bool train) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDropout"));
    return (*func)(ctx, out, mask, input, p, train);
}

DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask,
                                       double p, bool train) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDropoutInp"));
    return (*func)(ctx, input, mask, p, train);
}

DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMSELoss"));
    return (*func)(ctx, out, input, target, reduction);
}

DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMSELossBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, reduction);
}

DIOPI_API diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
                                             diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, float, float, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLoss"));
    return (*func)(ctx, out, inputs, targets, alpha, gamma, reduction);
}

DIOPI_API diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                                     diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiTensorHandle_t, float, float, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidFocalLossBackward"));
    return (*func)(ctx, grad_output, input, target, grad_input, gamma, alpha, reduction);
}

DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                             diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                             int64_t ignore_index, double label_smoothing) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t,
        int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCrossEntropyLoss"));
    return (*func)(ctx, out, input, target, weight, reduction, ignore_index, label_smoothing);
}

DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCrossEntropyLossBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
}

DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                    int64_t ignore_index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t,
        int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNLLLoss"));
    return (*func)(ctx, out, input, target, weight, reduction, ignore_index);
}

DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNLLLossBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index);
}

DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCEWithLogits"));
    return (*func)(ctx, out, input, target, weight, pos_weight, reduction);
}

DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCEWithLogitsBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, weight, pos_weight, reduction);
}

DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCELoss"));
    return (*func)(ctx, out, input, target, weight, reduction);
}

DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBCELossBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, weight, reduction);
}

DIOPI_API diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSign"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAbsInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAbs"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNegInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeg"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFloorInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFloor"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSqrtInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSqrt"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSinInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSin"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCosInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCos"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanhInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanh"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                         diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTanhBackward"));
    return (*func)(ctx, grad_input, grad_output, output);
}

DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoid"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSigmoidBackward"));
    return (*func)(ctx, grad_input, grad_output, output);
}

DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExpInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExp"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog2Inp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog2"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog10Inp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLog10"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErfInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiErf"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowScalar"));
    return (*func)(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPow"));
    return (*func)(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPowTensor"));
    return (*func)(ctx, out, input, exponent);
}

DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdd"));
    return (*func)(ctx, out, input, other, alpha);
}

DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddScalar"));
    return (*func)(ctx, out, input, other, alpha);
}

DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSub"));
    return (*func)(ctx, out, input, other, alpha);
}

DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSubScalar"));
    return (*func)(ctx, out, input, other, alpha);
}

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMul"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMulScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDiv"));
    return (*func)(ctx, out, input, other, rounding_mode);
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*, diopiRoundMode_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiDivScalar"));
    return (*func)(ctx, out, input, other, rounding_mode);
}

DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBmm"));
    return (*func)(ctx, out, input, mat2);
}

DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcmul"));
    return (*func)(ctx, out, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMatmul"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddcdiv"));
    return (*func)(ctx, out, input, tensor1, tensor2, value);
}

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAddmm"));
    return (*func)(ctx, out, input, mat1, mat2, beta, alpha);
}

DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampInpScalar"));
    return (*func)(ctx, input, min, max);
}

DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampInp"));
    return (*func)(ctx, input, min, max);
}

DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampScalar"));
    return (*func)(ctx, out, input, min, max);
}

DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClamp"));
    return (*func)(ctx, out, input, min, max);
}

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxInpScalar"));
    return (*func)(ctx, input, max);
}

DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxInp"));
    return (*func)(ctx, input, max);
}

DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMaxScalar"));
    return (*func)(ctx, out, input, max);
}

DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMax"));
    return (*func)(ctx, out, input, max);
}

DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinInpScalar"));
    return (*func)(ctx, input, min);
}

DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinInp"));
    return (*func)(ctx, input, min);
}

DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMinScalar"));
    return (*func)(ctx, out, input, min);
}

DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClampMin"));
    return (*func)(ctx, out, input, min);
}

DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiFill"));
    return (*func)(ctx, input, value);
}

DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAnd"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseAndScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOr"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseOrScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEqScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEq"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNeScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNe"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGeScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGe"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGtScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGt"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLeScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLe"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLtScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLt"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMean"));
    return (*func)(ctx, out, input, dim, dtype);
}

DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                 diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSum"));
    return (*func)(ctx, out, input, dim, dtype);
}

DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStd"));
    return (*func)(ctx, out, input, dim, unbiased);
}

DIOPI_API diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
                                diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMin"));
    return (*func)(ctx, min, min_indices, input, dim);
}

DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
                                diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMax"));
    return (*func)(ctx, max, max_indices, input, dim);
}

DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAny"));
    return (*func)(ctx, out, input, dim);
}

DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAll"));
    return (*func)(ctx, out, input, dim);
}

DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmax"));
    return (*func)(ctx, out, input, dim, dtype);
}

DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSoftmaxBackward"));
    return (*func)(ctx, grad_input, grad_output, output, dim, input_dtype);
}

DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogSoftmax"));
    return (*func)(ctx, out, input, dim, dtype);
}

DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLogSoftmaxBackward"));
    return (*func)(ctx, grad_input, grad_output, output, dim, input_dtype);
}

DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t* indices, int64_t nums) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndex"));
    return (*func)(ctx, out, input, indices, nums);
}

DIOPI_API diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                          diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexBackward"));
    return (*func)(ctx, grad_input, zeros_like_input, indices, nums, grad);
}

DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexSelect"));
    return (*func)(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                                diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexSelectBackward"));
    return (*func)(ctx, grad_input, grad, input_sizes, dim, index);
}

DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx,  diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    diopiError_t (*func) (diopiContextHandle_t,  diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelect"));
    return (*func)(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                           diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelectBackward"));
    return (*func)(ctx, grad_input, grad_output, input_sizes, dim, index);
}

DIOPI_API diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t src, int64_t dim, int64_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSelectScatter"));
    return (*func)(ctx, out, input, src, dim, index);
}

DIOPI_API diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                         diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSliceScatter"));
    return (*func)(ctx, out, input, src, dim, start, end, step);
}

DIOPI_API diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input,
                                  int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSlice"));
    return (*func)(ctx, null_out, input, dim, start, end, step);
}

DIOPI_API diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSliceBackward"));
    return (*func)(ctx, grad_input, grad_output, input_sizes, dim, start, end, step);
}

DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedScatter"));
    return (*func)(ctx, out, input, mask, source);
}

DIOPI_API diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
                                diopiConstTensorHandle_t scores, double iou_threshold) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNms"));
    return (*func)(ctx, out, dets, scores, iou_threshold);
}

DIOPI_API diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNonzero"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinear"));
    return (*func)(ctx, out, input, weight, bias);
}

DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                           diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinearBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight);
}

DIOPI_API diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                     int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double, int64_t,
        int64_t, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlign"));
    return (*func)(ctx, out, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
}

DIOPI_API diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                             diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height,
                                             int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height,
                                             int64_t width, int64_t sampling_ratio, bool aligned) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, double, int64_t,
        int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoiAlignBackward"));
    return (*func)(ctx, out, grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
}

DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf,
                                double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        double, double, double, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSgd"));
    return (*func)(ctx, w, dw, buf, lr, momentum, dampening, weight_decay, nesterov);
}

DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t *parameters,
                                         int64_t num_parameters, double max_norm, double norm_type, bool error_if_nonfinite) {
    diopiError_t (*func) (diopiContextHandle_t, double*, diopiTensorHandle_t *,
        int64_t, double, double, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiClipGradNorm"));
    return (*func)(ctx, out, parameters, num_parameters, max_norm, norm_type, error_if_nonfinite);
}

DIOPI_API diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout,
                                             diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, double, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbeddingRenorm_"));
    return (*func)(ctx, inout, indices, max_norm, norm_type);
}

DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbedding"));
    return (*func)(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
}

DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiEmbeddingBackward"));
    return (*func)(ctx, out, grad, indices, num_weights, padding_idx, scale_grad_byfreq, sparse);
}

DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTril"));
    return (*func)(ctx, out, input, diagonal);
}

DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t*, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCat"));
    return (*func)(ctx, out, tensors, num_inputs, dim);
}

DIOPI_API diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs,
                                           diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, int64_t,
        diopiConstTensorHandle_t, const diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSplitWithSizes"));
    return (*func)(ctx, outs, num_outs, input, splitSizes, dim);
}

DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                  diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t*, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiStack"));
    return (*func)(ctx, out, tensors, numTensors, dim);
}

DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, bool, const bool*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSort"));
    return (*func)(ctx, values, indices, input, dim, descending, stable);
}

DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t, bool, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTopk"));
    return (*func)(ctx, values, indices, input, k, dim, largest, sorted);
}

DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                      diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiTranspose"));
    return (*func)(ctx, out, input, dim0, dim1);
}

DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input, int64_t num_classes) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiOneHot"));
    return (*func)(ctx, out, input, num_classes);
}

DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiWhere"));
    return (*func)(ctx, out, condition, input, other);
}

DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFill"));
    return (*func)(ctx, out, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillInp"));
    return (*func)(ctx, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillScalar"));
    return (*func)(ctx, out, input, mask, value);
}

DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedFillInpScalar"));
    return (*func)(ctx, input, mask, value);
}

DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReciprocal"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiReciprocalInp"));
    return (*func)(ctx, input);
}

DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                                  diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                                  float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        float, float, float, float, float, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdamW"));
    return (*func)(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

DIOPI_API diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, 
                                            diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvTranspose2d"));
    return (*func)(ctx, out, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnfold"));
    return (*func)(ctx, out, input, dim, size, step);
}

DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, int64_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnfoldBackward"));
    return (*func)(ctx, grad_input, grad_output, input_sizes, dim, size, step);
}

DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCumsum"));
    return (*func)(ctx, out, input, dim, dtype);
}

DIOPI_API diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                  double p, const int64_t* compute_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        double, const int64_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCdist"));
    return (*func)(ctx, out, input1, input2, p, compute_mode);
}

DIOPI_API diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, double, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCdistBackward"));
    return (*func)(ctx, grad_input, grad_output, input1, input2, p, cdist);
}

DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBitwiseNot"));
    return (*func)(ctx, out, input);
}

DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const int64_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiArgmax"));
    return (*func)(ctx, out, input, dim, keepdim);
}

DIOPI_API diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                     diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, float, float, float, float);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdadelta"));
    return (*func)(ctx, input, grad, square_avg, acc_delta, lr, rho, eps, weight_decay);
}

DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq,
                                 diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, float, float, float, float, float, int64_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdam"));
    return (*func)(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

DIOPI_API diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiReduction_t reduction, double beta) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiReduction_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSmoothL1Loss"));
    return (*func)(ctx, out, input, target, reduction, beta);
}

DIOPI_API diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiReduction_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiSmoothL1LossBackward"));
    return (*func)(ctx, grad_input, grad_output, input, target, reduction, beta);
}

DIOPI_API diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                          diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        diopiSize_t, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution3d"));
    return (*func)(ctx, out, input, weight, bias, stride, padding, dilation, groups);
}

DIOPI_API diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t *, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiSize_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiConvolution3dBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

DIOPI_API diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3d"));
    return (*func)(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
}

DIOPI_API diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                 diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                                                 diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t,
        diopiSize_t, diopiSize_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3dWithIndices"));
    return (*func)(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}

DIOPI_API diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t, diopiSize_t, diopiSize_t,
        diopiSize_t, bool, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaxPool3dBackward"));
    return (*func)(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool3d"));
    return (*func)(ctx, out, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                      diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveAvgPool3dBackward"));
    return (*func)(ctx, grad_input, grad_output, input);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3d"));
    return (*func)(ctx, out, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                              diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3dWithIndices"));
    return (*func)(ctx, out, indices, input, output_size);
}

DIOPI_API diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiAdaptiveMaxPool3dBackward"));
    return (*func)(ctx, grad_input, grad_output, input, indices);
}

DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedSelect"));
    return (*func)(ctx, out, input, mask);
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                                 diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaskedSelectBackward"));
    return (*func)(ctx, grad_input, grad_output, input, mask);
}

DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMaximum"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMinimum"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiMm"));
    return (*func)(ctx, out, input, mat2);
}

DIOPI_API diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                            int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillScalar"));
    return (*func)(ctx, out, input, dim, index, value);
}

DIOPI_API diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                      int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFill"));
    return (*func)(ctx, out, input, dim, index, value);
}

DIOPI_API diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                               int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillInpScalar"));
    return (*func)(ctx, input, dim, index, value);
}

DIOPI_API diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                         int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexFillInp"));
    return (*func)(ctx, input, dim, index, value);
}

DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiExpand"));
    return (*func)(ctx, out, input, size);
}

DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLinspace"));
    return (*func)(ctx, out, start, end, steps);
}

DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPermute"));
    return (*func)(ctx, out, input, dims);
}

DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, double* value) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t, const char*, double*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiPad"));
    return (*func)(ctx, out, input, pad, mode, value);
}

DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRoll"));
    return (*func)(ctx, out, input, shifts, dims);
}

DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim, diopiDtype_t dtype) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, const diopiScalar_t*, diopiSize_t, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiNorm"));
    return (*func)(ctx, out, input, p, dim, dtype);
}

DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, int64_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupNorm"));
    return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, num_groups, eps);
}

DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, int64_t num_groups) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGroupNormBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, mean, rstd, num_groups);
}

DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, int64_t* dim,
                                   bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t*, diopiConstTensorHandle_t, int64_t*,
        bool, bool, diopiTensorHandle_t, diopiTensorHandle_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUnique"));
    return (*func)(ctx, out, input, dim, sorted, return_counts, indices, counts);
}

DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t* dim, diopiDtype_t type) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t*, diopiDtype_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiProd"));
    return (*func)(ctx, out, input, dim, type);
}

DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, 
                                    diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                    diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiReduction_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCTCLoss"));
    return (*func)(ctx, out, neg_log_likelihood, log_alpha, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets,
                                            diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha,
                                            int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        int64_t, diopiReduction_t, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCTCLossBackward"));
    return (*func)(ctx, grad_input, grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, reduction, zero_infinity);
}

DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainderTensor"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiScalar_t* other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainderScalar"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiScalar_t* input, diopiConstTensorHandle_t other) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiScalar_t*, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRemainder"));
    return (*func)(ctx, out, input, other);
}

DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGather"));
    return (*func)(ctx, out, input, dim, index);
}

DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiGatherBackward"));
    return (*func)(ctx, grad_input, grad_output, input, dim, index);
}

DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterInp"));
    return (*func)(ctx, input, dim, src, index, reduce);
}

DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, int64_t, const diopiScalar_t*, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterInpScalar"));
    return (*func)(ctx, input, dim, value, index, reduce);
}

DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatter"));
    return (*func)(ctx, out, input, dim, src, index, reduce);
}

DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t, const diopiScalar_t*, diopiConstTensorHandle_t, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiScatterScalar"));
    return (*func)(ctx, out, input, dim, value, index, reduce);
}

DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, bool accumulate) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexPutInp"));
    return (*func)(ctx, input, values, indices, accumulate);
}

DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, bool accumulate) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t*, bool);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiIndexPut"));
    return (*func)(ctx, out, input, values, indices, accumulate);
}

DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, int64_t, const int64_t*, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRandomInp"));
    return (*func)(ctx, inout, from, to, idx);
}

DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, double, double, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUniformInp"));
    return (*func)(ctx, inout, from, to, idx);
}

DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulli"));
    return (*func)(ctx, out, input, idx);
}

DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulliInp"));
    return (*func)(ctx, inout, idx);
}

DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, double, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiBernoulliScalar"));
    return (*func)(ctx, out, p, idx);
}

DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, const diopiScalar_t*, const diopiScalar_t*, const diopiScalar_t*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiArange"));
    return (*func)(ctx, out, start, end, step);
}

DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, int64_t, int64_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiRandperm"));
    return (*func)(ctx, out, n, idx);
}

DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, double);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLayerNorm"));
    return (*func)(ctx, out, save_mean, save_invstd, input, weight, bias, normalized_shape, eps);
}

DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                              diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                              diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiTensorHandle_t, diopiTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiConstTensorHandle_t,
        diopiConstTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiLayerNormBackward"));
    return (*func)(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, mean, rstd, normalized_shape);
}

DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    diopiError_t (*func) (diopiContextHandle_t, diopiConstTensorHandle_t, diopiTensorHandle_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiCopyInp"));
    return (*func)(ctx, src, input);
}

DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleNearest"));
    return (*func)(ctx, out, input, size);
}

DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleNearestBackward"));
    return (*func)(ctx, grad_input, grad_output, out_size, in_size);
}

DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char* mode) {
    diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t, diopiConstTensorHandle_t, diopiSize_t,
        bool, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleLinear"));
    return (*func)(ctx, out, input, size, align_corners, mode);
}

DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx,  diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
    diopiError_t (*func) (diopiContextHandle_t,  diopiTensorHandle_t, diopiConstTensorHandle_t,
        diopiSize_t, diopiSize_t, bool, const char*);
    func = reinterpret_cast<decltype(func)>(dlsym(handle, "diopiUpsampleLinearBackward"));
    return (*func)(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
}
