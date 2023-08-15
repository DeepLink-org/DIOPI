/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

//NOLINTBEGIN
#include <pybind11/pybind11.h>
#include "litert.hpp"
#ifdef TEST_USE_ADAPTOR
#include <diopi/diopi_adaptors.hpp>
#endif
#include <diopi/diopirt.h>
namespace py = pybind11;

PYBIND11_MODULE(diopi_functions, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring
    m.def("diopiGetVendorName", &diopiGetVendorName);
    m.def("diopiGetImplVersion", &diopiGetImplVersion);
    m.def("diopiGetVersion", &diopiGetVersion);
    m.def("diopiGetLastErrorString", &diopiGetLastErrorString);
    m.def("diopiConvolution2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution2d) {
    
            diopiError_t ret = diopiadaptor::diopiConvolution2d(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
        if (diopiConvolution2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiConvolution2dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
        if (diopiBatchNorm) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNorm(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormStats", [](diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, double eps) {
        if (diopiBatchNormStats) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormStats(ctx, mean, invstd, input, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormGatherStatsWithCounts", [](diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean_all, diopiConstTensorHandle_t invstd_all, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, float momentum, float eps, diopiConstTensorHandle_t counts) {
        if (diopiBatchNormGatherStatsWithCounts) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormGatherStatsWithCounts(ctx, mean, invstd, input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackwardReduce", [](diopiContextHandle_t ctx, diopiTensorHandle_t sum_dy, diopiTensorHandle_t sum_dy_xmu, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, bool input_g, bool weight_g, bool bias_g) {
        if (diopiBatchNormBackwardReduce) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormBackwardReduce(ctx, sum_dy, sum_dy_xmu, grad_weight, grad_bias, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackwardElemt", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t sum_dy, diopiConstTensorHandle_t sum_dy_xmu, diopiConstTensorHandle_t count) {
        if (diopiBatchNormBackwardElemt) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormBackwardElemt(ctx, grad_input, grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormElemt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps) {
        if (diopiBatchNormElemt) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormElemt(ctx, out, input, weight, bias, mean, invstd, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
        if (diopiBatchNormBackward) {
    
            diopiError_t ret = diopiadaptor::diopiBatchNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiRelu) {
    
            diopiError_t ret = diopiadaptor::diopiRelu(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiReluInp) {
    
            diopiError_t ret = diopiadaptor::diopiReluInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanh", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanh) {
    
            diopiError_t ret = diopiadaptor::diopiHardtanh(ctx, out, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanhInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanhInp) {
    
            diopiError_t ret = diopiadaptor::diopiHardtanhInp(ctx, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanhBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanhBackward) {
    
            diopiError_t ret = diopiadaptor::diopiHardtanhBackward(ctx, grad_input, grad_output, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswish", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiHardswish) {
    
            diopiError_t ret = diopiadaptor::diopiHardswish(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswishInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiHardswishInp) {
    
            diopiError_t ret = diopiadaptor::diopiHardswishInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswishBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiHardswishBackward) {
    
            diopiError_t ret = diopiadaptor::diopiHardswishBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThreshold", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
        if (diopiThreshold) {
    
            diopiError_t ret = diopiadaptor::diopiThreshold(ctx, out, input, threshold, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThresholdInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
        if (diopiThresholdInp) {
    
            diopiError_t ret = diopiadaptor::diopiThresholdInp(ctx, input, threshold, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThresholdBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
        if (diopiThresholdBackward) {
    
            diopiError_t ret = diopiadaptor::diopiThresholdBackward(ctx, grad_input, grad_output, input, threshold);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
        if (diopiGelu) {
    
            diopiError_t ret = diopiadaptor::diopiGelu(ctx, out, input, approximate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const char* approximate) {
        if (diopiGeluBackward) {
    
            diopiError_t ret = diopiadaptor::diopiGeluBackward(ctx, grad_input, grad_output, input, approximate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyRelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
        if (diopiLeakyRelu) {
    
            diopiError_t ret = diopiadaptor::diopiLeakyRelu(ctx, out, input, negative_slope);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyReluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
        if (diopiLeakyReluInp) {
    
            diopiError_t ret = diopiadaptor::diopiLeakyReluInp(ctx, input, negative_slope);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyReluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
        if (diopiLeakyReluBackward) {
    
            diopiError_t ret = diopiadaptor::diopiLeakyReluBackward(ctx, grad_input, grad_output, input, negative_slope, input_is_result);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
        if (diopiAvgPool2d) {
    
            diopiError_t ret = diopiadaptor::diopiAvgPool2d(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
        if (diopiAvgPool2d) {
    
            diopiError_t ret = diopiadaptor::diopiAvgPool2d(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
        if (diopiAvgPool2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAvgPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
        if (diopiAvgPool2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAvgPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool2d) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool2d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool2dWithIndices) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool2dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
        if (diopiMaxPool2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveAvgPool2d) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveAvgPool2d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiAdaptiveAvgPool2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveAvgPool2dBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool2d) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool2d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool2dWithIndices) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool2dWithIndices(ctx, out, indices, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
        if (diopiAdaptiveMaxPool2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool2dBackward(ctx, grad_input, grad_output, input, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDropout", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train) {
        if (diopiDropout) {
    
            diopiError_t ret = diopiadaptor::diopiDropout(ctx, out, mask, input, p, train);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDropoutInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {
        if (diopiDropoutInp) {
    
            diopiError_t ret = diopiadaptor::diopiDropoutInp(ctx, input, mask, p, train);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMSELoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
        if (diopiMSELoss) {
    
            diopiError_t ret = diopiadaptor::diopiMSELoss(ctx, out, input, target, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMSELossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
        if (diopiMSELossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiMSELossBackward(ctx, grad_input, grad_output, input, target, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidFocalLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
        if (diopiSigmoidFocalLoss) {
    
            diopiError_t ret = diopiadaptor::diopiSigmoidFocalLoss(ctx, out, inputs, targets, alpha, gamma, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidFocalLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
        if (diopiSigmoidFocalLossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSigmoidFocalLossBackward(ctx, grad_output, input, target, grad_input, gamma, alpha, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCrossEntropyLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
        if (diopiCrossEntropyLoss) {
    
            diopiError_t ret = diopiadaptor::diopiCrossEntropyLoss(ctx, out, input, target, weight, reduction, ignore_index, label_smoothing);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCrossEntropyLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
        if (diopiCrossEntropyLossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiCrossEntropyLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNLLLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
        if (diopiNLLLoss) {
    
            diopiError_t ret = diopiadaptor::diopiNLLLoss(ctx, out, input, target, weight, reduction, ignore_index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNLLLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
        if (diopiNLLLossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiNLLLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCEWithLogits", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
        if (diopiBCEWithLogits) {
    
            diopiError_t ret = diopiadaptor::diopiBCEWithLogits(ctx, out, input, target, weight, pos_weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCEWithLogitsBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
        if (diopiBCEWithLogitsBackward) {
    
            diopiError_t ret = diopiadaptor::diopiBCEWithLogitsBackward(ctx, grad_input, grad_output, input, target, weight, pos_weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCELoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
        if (diopiBCELoss) {
    
            diopiError_t ret = diopiadaptor::diopiBCELoss(ctx, out, input, target, weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCELossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
        if (diopiBCELossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiBCELossBackward(ctx, grad_input, grad_output, input, target, weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSign", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSign) {
    
            diopiError_t ret = diopiadaptor::diopiSign(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAbsInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAbsInp) {
    
            diopiError_t ret = diopiadaptor::diopiAbsInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAbs", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAbs) {
    
            diopiError_t ret = diopiadaptor::diopiAbs(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNegInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiNegInp) {
    
            diopiError_t ret = diopiadaptor::diopiNegInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeg", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiNeg) {
    
            diopiError_t ret = diopiadaptor::diopiNeg(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFloorInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiFloorInp) {
    
            diopiError_t ret = diopiadaptor::diopiFloorInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFloor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiFloor) {
    
            diopiError_t ret = diopiadaptor::diopiFloor(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCeilInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiCeilInp) {
    
            diopiError_t ret = diopiadaptor::diopiCeilInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCeil", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiCeil) {
    
            diopiError_t ret = diopiadaptor::diopiCeil(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSqrtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSqrtInp) {
    
            diopiError_t ret = diopiadaptor::diopiSqrtInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSqrt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSqrt) {
    
            diopiError_t ret = diopiadaptor::diopiSqrt(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRsqrtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiRsqrtInp) {
    
            diopiError_t ret = diopiadaptor::diopiRsqrtInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRsqrt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiRsqrt) {
    
            diopiError_t ret = diopiadaptor::diopiRsqrt(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSinInp) {
    
            diopiError_t ret = diopiadaptor::diopiSinInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSin) {
    
            diopiError_t ret = diopiadaptor::diopiSin(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAsinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAsinInp) {
    
            diopiError_t ret = diopiadaptor::diopiAsinInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAsin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAsin) {
    
            diopiError_t ret = diopiadaptor::diopiAsin(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCosInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiCosInp) {
    
            diopiError_t ret = diopiadaptor::diopiCosInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCos", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiCos) {
    
            diopiError_t ret = diopiadaptor::diopiCos(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanhInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiTanhInp) {
    
            diopiError_t ret = diopiadaptor::diopiTanhInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanh", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiTanh) {
    
            diopiError_t ret = diopiadaptor::diopiTanh(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanhBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
        if (diopiTanhBackward) {
    
            diopiError_t ret = diopiadaptor::diopiTanhBackward(ctx, grad_input, grad_output, output);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAtan", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAtan) {
    
            diopiError_t ret = diopiadaptor::diopiAtan(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAtanInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAtanInp) {
    
            diopiError_t ret = diopiadaptor::diopiAtanInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSigmoidInp) {
    
            diopiError_t ret = diopiadaptor::diopiSigmoidInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoid", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSigmoid) {
    
            diopiError_t ret = diopiadaptor::diopiSigmoid(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
        if (diopiSigmoidBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSigmoidBackward(ctx, grad_input, grad_output, output);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSiluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSiluInp) {
    
            diopiError_t ret = diopiadaptor::diopiSiluInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSilu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSilu) {
    
            diopiError_t ret = diopiadaptor::diopiSilu(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSiluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiSiluBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSiluBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExpInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiExpInp) {
    
            diopiError_t ret = diopiadaptor::diopiExpInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExp", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiExp) {
    
            diopiError_t ret = diopiadaptor::diopiExp(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLogInp) {
    
            diopiError_t ret = diopiadaptor::diopiLogInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog) {
    
            diopiError_t ret = diopiadaptor::diopiLog(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog2Inp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLog2Inp) {
    
            diopiError_t ret = diopiadaptor::diopiLog2Inp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog2", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog2) {
    
            diopiError_t ret = diopiadaptor::diopiLog2(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog10Inp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLog10Inp) {
    
            diopiError_t ret = diopiadaptor::diopiLog10Inp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog10", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog10) {
    
            diopiError_t ret = diopiadaptor::diopiLog10(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiErfInp) {
    
            diopiError_t ret = diopiadaptor::diopiErfInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErf", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiErf) {
    
            diopiError_t ret = diopiadaptor::diopiErf(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
        if (diopiPowScalar) {
    
            diopiError_t ret = diopiadaptor::diopiPowScalar(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPow", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
        if (diopiPow) {
    
            diopiError_t ret = diopiadaptor::diopiPow(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
        if (diopiPowInp) {
    
            diopiError_t ret = diopiadaptor::diopiPowInp(ctx, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
        if (diopiPowTensor) {
    
            diopiError_t ret = diopiadaptor::diopiPowTensor(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowInpTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
        if (diopiPowInpTensor) {
    
            diopiError_t ret = diopiadaptor::diopiPowInpTensor(ctx, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiAdd) {
    
            diopiError_t ret = diopiadaptor::diopiAdd(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiAddInp) {
    
            diopiError_t ret = diopiadaptor::diopiAddInp(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiAddScalar) {
    
            diopiError_t ret = diopiadaptor::diopiAddScalar(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiAddInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiAddInpScalar(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSub", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiSub) {
    
            diopiError_t ret = diopiadaptor::diopiSub(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiSubInp) {
    
            diopiError_t ret = diopiadaptor::diopiSubInp(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiSubScalar) {
    
            diopiError_t ret = diopiadaptor::diopiSubScalar(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiSubInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiSubInpScalar(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMul) {
    
            diopiError_t ret = diopiadaptor::diopiMul(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMulInp) {
    
            diopiError_t ret = diopiadaptor::diopiMulInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiMulScalar) {
    
            diopiError_t ret = diopiadaptor::diopiMulScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiMulInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiMulInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDiv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
        if (diopiDiv) {
    
            diopiError_t ret = diopiadaptor::diopiDiv(ctx, out, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
        if (diopiDivInp) {
    
            diopiError_t ret = diopiadaptor::diopiDivInp(ctx, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
        if (diopiDivScalar) {
    
            diopiError_t ret = diopiadaptor::diopiDivScalar(ctx, out, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
        if (diopiDivInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiDivInpScalar(ctx, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
        if (diopiBmm) {
    
            diopiError_t ret = diopiadaptor::diopiBmm(ctx, out, input, mat2);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBaddbmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
        if (diopiBaddbmm) {
    
            diopiError_t ret = diopiadaptor::diopiBaddbmm(ctx, out, input, batch1, batch2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBaddbmmInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
        if (diopiBaddbmmInp) {
    
            diopiError_t ret = diopiadaptor::diopiBaddbmmInp(ctx, input, batch1, batch2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcmul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcmul) {
    
            diopiError_t ret = diopiadaptor::diopiAddcmul(ctx, out, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcmulInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcmulInp) {
    
            diopiError_t ret = diopiadaptor::diopiAddcmulInp(ctx, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMatmul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMatmul) {
    
            diopiError_t ret = diopiadaptor::diopiMatmul(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcdiv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcdiv) {
    
            diopiError_t ret = diopiadaptor::diopiAddcdiv(ctx, out, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcdivInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcdivInp) {
    
            diopiError_t ret = diopiadaptor::diopiAddcdivInp(ctx, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
        if (diopiAddmm) {
    
            diopiError_t ret = diopiadaptor::diopiAddmm(ctx, out, input, mat1, mat2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCholesky", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror) {
        if (diopiCholesky) {
    
            diopiError_t ret = diopiadaptor::diopiCholesky(ctx, out, info, mat, upper, checkerror);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCholeskyBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper) {
        if (diopiCholeskyBackward) {
    
            diopiError_t ret = diopiadaptor::diopiCholeskyBackward(ctx, grad_mat, grad_output, L, upper);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriangularSolve", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
        if (diopiTriangularSolve) {
    
            diopiError_t ret = diopiadaptor::diopiTriangularSolve(ctx, out, cloned_mat, b, mat, upper, transpose, unitriangular);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriangularSolveBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
        if (diopiTriangularSolveBackward) {
    
            diopiError_t ret = diopiadaptor::diopiTriangularSolveBackward(ctx, grad_b, grad_mat, grad_x, grad_cloned_mat, x, b, mat, upper, transpose, unitriangular);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
        if (diopiClampInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampInpScalar(ctx, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
        if (diopiClampInp) {
    
            diopiError_t ret = diopiadaptor::diopiClampInp(ctx, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
        if (diopiClampScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampScalar(ctx, out, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClamp", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
        if (diopiClamp) {
    
            diopiError_t ret = diopiadaptor::diopiClamp(ctx, out, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
        if (diopiClampMaxInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampMaxInpScalar(ctx, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
        if (diopiClampMaxInp) {
    
            diopiError_t ret = diopiadaptor::diopiClampMaxInp(ctx, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
        if (diopiClampMaxScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampMaxScalar(ctx, out, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
        if (diopiClampMax) {
    
            diopiError_t ret = diopiadaptor::diopiClampMax(ctx, out, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
        if (diopiClampMinInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampMinInpScalar(ctx, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
        if (diopiClampMinInp) {
    
            diopiError_t ret = diopiadaptor::diopiClampMinInp(ctx, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
        if (diopiClampMinScalar) {
    
            diopiError_t ret = diopiadaptor::diopiClampMinScalar(ctx, out, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
        if (diopiClampMin) {
    
            diopiError_t ret = diopiadaptor::diopiClampMin(ctx, out, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
        if (diopiFill) {
    
            diopiError_t ret = diopiadaptor::diopiFill(ctx, input, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalAnd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalAnd) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalAnd(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalAndInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalAndInp) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalAndInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalOr", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalOr) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalOr(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalOrInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalOrInp) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalOrInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalNot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLogicalNot) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalNot(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalNotInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLogicalNotInp) {
    
            diopiError_t ret = diopiadaptor::diopiLogicalNotInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAnd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseAnd) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseAnd(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseAndInp) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseAndInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseAndScalar) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseAndScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseAndInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseAndInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOr", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseOr) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseOr(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseOrInp) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseOrInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseOrScalar) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseOrScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseOrInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseOrInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseNot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiBitwiseNot) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseNot(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseNotInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiBitwiseNotInp) {
    
            diopiError_t ret = diopiadaptor::diopiBitwiseNotInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiEqScalar) {
    
            diopiError_t ret = diopiadaptor::diopiEqScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiEqInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiEqInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEq", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiEq) {
    
            diopiError_t ret = diopiadaptor::diopiEq(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiEqInp) {
    
            diopiError_t ret = diopiadaptor::diopiEqInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiNeScalar) {
    
            diopiError_t ret = diopiadaptor::diopiNeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiNeInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiNeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiNe) {
    
            diopiError_t ret = diopiadaptor::diopiNe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiNeInp) {
    
            diopiError_t ret = diopiadaptor::diopiNeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGeScalar) {
    
            diopiError_t ret = diopiadaptor::diopiGeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGeInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiGeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGe) {
    
            diopiError_t ret = diopiadaptor::diopiGe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGeInp) {
    
            diopiError_t ret = diopiadaptor::diopiGeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGtScalar) {
    
            diopiError_t ret = diopiadaptor::diopiGtScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGtInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiGtInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGt) {
    
            diopiError_t ret = diopiadaptor::diopiGt(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGtInp) {
    
            diopiError_t ret = diopiadaptor::diopiGtInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLeScalar) {
    
            diopiError_t ret = diopiadaptor::diopiLeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLeInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiLeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLe) {
    
            diopiError_t ret = diopiadaptor::diopiLe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLeInp) {
    
            diopiError_t ret = diopiadaptor::diopiLeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLtScalar) {
    
            diopiError_t ret = diopiadaptor::diopiLtScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLtInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiLtInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLt) {
    
            diopiError_t ret = diopiadaptor::diopiLt(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLtInp) {
    
            diopiError_t ret = diopiadaptor::diopiLtInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMean", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
        if (diopiMean) {
    
            diopiError_t ret = diopiadaptor::diopiMean(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
        if (diopiSum) {
    
            diopiError_t ret = diopiadaptor::diopiSum(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiStd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
        if (diopiStd) {
    
            diopiError_t ret = diopiadaptor::diopiStd(ctx, out, input, dim, unbiased);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMin", [](diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiMin) {
    
            diopiError_t ret = diopiadaptor::diopiMin(ctx, min, min_indices, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMinAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
        if (diopiMinAll) {
    
            diopiError_t ret = diopiadaptor::diopiMinAll(ctx, min, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMax", [](diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiMax) {
    
            diopiError_t ret = diopiadaptor::diopiMax(ctx, max, max_indices, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
        if (diopiMaxAll) {
    
            diopiError_t ret = diopiadaptor::diopiMaxAll(ctx, max, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAny", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
        if (diopiAny) {
    
            diopiError_t ret = diopiadaptor::diopiAny(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAny", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAny) {
    
            diopiError_t ret = diopiadaptor::diopiAny(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
        if (diopiAll) {
    
            diopiError_t ret = diopiadaptor::diopiAll(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAll) {
    
            diopiError_t ret = diopiadaptor::diopiAll(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSoftmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiSoftmax) {
    
            diopiError_t ret = diopiadaptor::diopiSoftmax(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSoftmaxBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
        if (diopiSoftmaxBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSoftmaxBackward(ctx, grad_input, grad_output, output, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogSoftmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiLogSoftmax) {
    
            diopiError_t ret = diopiadaptor::diopiLogSoftmax(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogSoftmaxBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
        if (diopiLogSoftmaxBackward) {
    
            diopiError_t ret = diopiadaptor::diopiLogSoftmaxBackward(ctx, grad_input, grad_output, output, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndex", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, py::list& indices, int64_t nums) {
        if (diopiIndex) {
            std::vector<diopiConstTensorHandle_t> indicesV(nums);
            for (int i = 0; i < nums; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiIndex(ctx, &outHandle, input, indicesDIOPI, nums);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input, py::list& indices, int64_t nums, diopiConstTensorHandle_t grad) {
        if (diopiIndexBackward) {
            std::vector<diopiConstTensorHandle_t> indicesV(nums);
            for (int i = 0; i < nums; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiadaptor::diopiIndexBackward(ctx, grad_input, zeros_like_input, indicesDIOPI, nums, grad);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexSelect", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiIndexSelect) {
    
            diopiError_t ret = diopiadaptor::diopiIndexSelect(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiIndexSelectBackward) {
    
            diopiError_t ret = diopiadaptor::diopiIndexSelectBackward(ctx, grad_input, grad, input_sizes, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelect", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
        if (diopiSelect) {
    
            diopiError_t ret = diopiadaptor::diopiSelect(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
        if (diopiSelectBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSelectBackward(ctx, grad_input, grad_output, input_sizes, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelectScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t index) {
        if (diopiSelectScatter) {
    
            diopiError_t ret = diopiadaptor::diopiSelectScatter(ctx, out, input, src, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSliceScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSliceScatter) {
    
            diopiError_t ret = diopiadaptor::diopiSliceScatter(ctx, out, input, src, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSlice", [](diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSlice) {
    
            diopiError_t ret = diopiadaptor::diopiSlice(ctx, null_out, input, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSliceBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSliceBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSliceBackward(ctx, grad_input, grad_output, input_sizes, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
        if (diopiMaskedScatter) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedScatter(ctx, out, input, mask, source);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNms", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores, double iou_threshold) {
        if (diopiNms) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiNms(ctx, &outHandle, dets, scores, iou_threshold);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNonzero", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input) {
        if (diopiNonzero) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiNonzero(ctx, &outHandle, input);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinear", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
        if (diopiLinear) {
    
            diopiError_t ret = diopiadaptor::diopiLinear(ctx, out, input, weight, bias);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinearBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
        if (diopiLinearBackward) {
    
            diopiError_t ret = diopiadaptor::diopiLinearBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoiAlign", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
        if (diopiRoiAlign) {
    
            diopiError_t ret = diopiadaptor::diopiRoiAlign(ctx, out, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoiAlignBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height, int64_t width, int64_t sampling_ratio, bool aligned) {
        if (diopiRoiAlignBackward) {
    
            diopiError_t ret = diopiadaptor::diopiRoiAlignBackward(ctx, out, grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgd", [](diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
        if (diopiSgd) {
    
            diopiError_t ret = diopiadaptor::diopiSgd(ctx, w, dw, buf, lr, momentum, dampening, weight_decay, nesterov);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClipGradNorm", [](diopiContextHandle_t ctx, void* out, py::list& grads, int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {
        if (diopiClipGradNorm) {
            std::vector<diopiTensorHandle_t> gradsV(num_grads);
            for (int i = 0; i < num_grads; ++i)
                gradsV[i] = grads[i].cast<PtrWrapper<diopiTensor>>().get();
            auto gradsDIOPI = gradsV.data();
            diopiError_t ret = diopiadaptor::diopiClipGradNorm(ctx, reinterpret_cast<double*>(out), gradsDIOPI, num_grads, max_norm, norm_type, error_if_nonfinite);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbeddingRenorm_", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
        if (diopiEmbeddingRenorm_) {
    
            diopiError_t ret = diopiadaptor::diopiEmbeddingRenorm_(ctx, inout, indices, max_norm, norm_type);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbedding", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
        if (diopiEmbedding) {
    
            diopiError_t ret = diopiadaptor::diopiEmbedding(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbeddingBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
        if (diopiEmbeddingBackward) {
    
            diopiError_t ret = diopiadaptor::diopiEmbeddingBackward(ctx, out, grad, indices, num_weights, padding_idx, scale_grad_byfreq, sparse);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTril", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
        if (diopiTril) {
    
            diopiError_t ret = diopiadaptor::diopiTril(ctx, out, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCat", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t num_inputs, int64_t dim) {
        if (diopiCat) {
            std::vector<diopiConstTensorHandle_t> tensorsV(num_inputs);
            for (int i = 0; i < num_inputs; ++i)
                tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
            auto tensorsDIOPI = tensorsV.data();
            diopiError_t ret = diopiadaptor::diopiCat(ctx, out, tensorsDIOPI, num_inputs, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSplitWithSizes", [](diopiContextHandle_t ctx, py::list& outs, int64_t num_outs, diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
        if (diopiSplitWithSizes) {
            std::vector<diopiTensorHandle_t> outsV(num_outs);
            for (int i = 0; i < num_outs; ++i)
                outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto outsDIOPI = outsV.data();
            diopiError_t ret = diopiadaptor::diopiSplitWithSizes(ctx, outsDIOPI, num_outs, input, splitSizes, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiStack", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t numTensors, int64_t dim) {
        if (diopiStack) {
            std::vector<diopiConstTensorHandle_t> tensorsV(numTensors);
            for (int i = 0; i < numTensors; ++i)
                tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
            auto tensorsDIOPI = tensorsV.data();
            diopiError_t ret = diopiadaptor::diopiStack(ctx, out, tensorsDIOPI, numTensors, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSort", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
        if (diopiSort) {
    
            diopiError_t ret = diopiadaptor::diopiSort(ctx, values, indices, input, dim, descending, stable);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSort", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending) {
        if (diopiSort) {
    
            diopiError_t ret = diopiadaptor::diopiSort(ctx, values, indices, input, dim, descending, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTopk", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
        if (diopiTopk) {
    
            diopiError_t ret = diopiadaptor::diopiTopk(ctx, values, indices, input, k, dim, largest, sorted);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTranspose", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
        if (diopiTranspose) {
    
            diopiError_t ret = diopiadaptor::diopiTranspose(ctx, out, input, dim0, dim1);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiOneHot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes) {
        if (diopiOneHot) {
    
            diopiError_t ret = diopiadaptor::diopiOneHot(ctx, out, input, num_classes);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiWhere", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiWhere) {
    
            diopiError_t ret = diopiadaptor::diopiWhere(ctx, out, condition, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
        if (diopiMaskedFill) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedFill(ctx, out, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
        if (diopiMaskedFillInp) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedFillInp(ctx, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
        if (diopiMaskedFillScalar) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedFillScalar(ctx, out, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
        if (diopiMaskedFillInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedFillInpScalar(ctx, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReciprocal", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiReciprocal) {
    
            diopiError_t ret = diopiadaptor::diopiReciprocal(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReciprocalInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiReciprocalInp) {
    
            diopiError_t ret = diopiadaptor::diopiReciprocalInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdamW", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
        if (diopiAdamW) {
    
            diopiError_t ret = diopiadaptor::diopiAdamW(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvTranspose2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
        if (diopiConvTranspose2d) {
    
            diopiError_t ret = diopiadaptor::diopiConvTranspose2d(ctx, out, input, weight, bias, stride, padding, output_padding, groups, dilation);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvTranspose2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, diopiSize_t output_padding, int64_t groups) {
        if (diopiConvTranspose2dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiConvTranspose2dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, output_padding, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnfold", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
        if (diopiUnfold) {
    
            diopiError_t ret = diopiadaptor::diopiUnfold(ctx, out, input, dim, size, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnfoldBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
        if (diopiUnfoldBackward) {
    
            diopiError_t ret = diopiadaptor::diopiUnfoldBackward(ctx, grad_input, grad_output, input_sizes, dim, size, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCumsum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiCumsum) {
    
            diopiError_t ret = diopiadaptor::diopiCumsum(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdist", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, const int64_t* compute_mode) {
        if (diopiCdist) {
    
            diopiError_t ret = diopiadaptor::diopiCdist(ctx, out, input1, input2, p, compute_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdist", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p) {
        if (diopiCdist) {
    
            diopiError_t ret = diopiadaptor::diopiCdist(ctx, out, input1, input2, p, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdistBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
        if (diopiCdistBackward) {
    
            diopiError_t ret = diopiadaptor::diopiCdistBackward(ctx, grad_input, grad_output, input1, input2, p, cdist);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArgmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
        if (diopiArgmax) {
    
            diopiError_t ret = diopiadaptor::diopiArgmax(ctx, out, input, dim, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArgmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, bool keepdim) {
        if (diopiArgmax) {
    
            diopiError_t ret = diopiadaptor::diopiArgmax(ctx, out, input, nullptr, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdadelta", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
        if (diopiAdadelta) {
    
            diopiError_t ret = diopiadaptor::diopiAdadelta(ctx, input, grad, square_avg, acc_delta, lr, rho, eps, weight_decay);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdam", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
        if (diopiAdam) {
    
            diopiError_t ret = diopiadaptor::diopiAdam(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRmsprop", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered) {
        if (diopiRmsprop) {
    
            diopiError_t ret = diopiadaptor::diopiRmsprop(ctx, input, grad, square_avg, grad_avg, momentum_buf, lr, alpha, eps, weight_decay, momentum, centered);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSmoothL1Loss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
        if (diopiSmoothL1Loss) {
    
            diopiError_t ret = diopiadaptor::diopiSmoothL1Loss(ctx, out, input, target, reduction, beta);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSmoothL1LossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
        if (diopiSmoothL1LossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiSmoothL1LossBackward(ctx, grad_input, grad_output, input, target, reduction, beta);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution3d) {
    
            diopiError_t ret = diopiadaptor::diopiConvolution3d(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
        if (diopiConvolution3dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiConvolution3dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool3d) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool3d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool3dWithIndices) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool3dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
        if (diopiMaxPool3dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiMaxPool3dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveAvgPool3d) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveAvgPool3d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiAdaptiveAvgPool3dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveAvgPool3dBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool3d) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool3d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool3dWithIndices) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool3dWithIndices(ctx, out, indices, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
        if (diopiAdaptiveMaxPool3dBackward) {
    
            diopiError_t ret = diopiadaptor::diopiAdaptiveMaxPool3dBackward(ctx, grad_input, grad_output, input, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedSelect", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
        if (diopiMaskedSelect) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiMaskedSelect(ctx, &outHandle, input, mask);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
        if (diopiMaskedSelectBackward) {
    
            diopiError_t ret = diopiadaptor::diopiMaskedSelectBackward(ctx, grad_input, grad_output, input, mask);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaximum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMaximum) {
    
            diopiError_t ret = diopiadaptor::diopiMaximum(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMinimum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMinimum) {
    
            diopiError_t ret = diopiadaptor::diopiMinimum(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
        if (diopiMm) {
    
            diopiError_t ret = diopiadaptor::diopiMm(ctx, out, input, mat2);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
        if (diopiIndexFillScalar) {
    
            diopiError_t ret = diopiadaptor::diopiIndexFillScalar(ctx, out, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
        if (diopiIndexFill) {
    
            diopiError_t ret = diopiadaptor::diopiIndexFill(ctx, out, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
        if (diopiIndexFillInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiIndexFillInpScalar(ctx, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
        if (diopiIndexFillInp) {
    
            diopiError_t ret = diopiadaptor::diopiIndexFillInp(ctx, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExpand", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiExpand) {
    
            diopiError_t ret = diopiadaptor::diopiExpand(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinspace", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
        if (diopiLinspace) {
    
            diopiError_t ret = diopiadaptor::diopiLinspace(ctx, out, start, end, steps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPermute", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
        if (diopiPermute) {
    
            diopiError_t ret = diopiadaptor::diopiPermute(ctx, out, input, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPad", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, const double* value) {
        if (diopiPad) {
    
            diopiError_t ret = diopiadaptor::diopiPad(ctx, out, input, pad, mode, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPad", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode) {
        if (diopiPad) {
    
            diopiError_t ret = diopiadaptor::diopiPad(ctx, out, input, pad, mode, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
        if (diopiRoll) {
    
            diopiError_t ret = diopiadaptor::diopiRoll(ctx, out, input, shifts, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFlip", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
        if (diopiFlip) {
    
            diopiError_t ret = diopiadaptor::diopiFlip(ctx, out, input, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
        if (diopiNorm) {
    
            diopiError_t ret = diopiadaptor::diopiNorm(ctx, out, input, p, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGroupNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
        if (diopiGroupNorm) {
    
            diopiError_t ret = diopiadaptor::diopiGroupNorm(ctx, out, save_mean, save_invstd, input, weight, bias, num_groups, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGroupNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t num_groups) {
        if (diopiGroupNormBackward) {
    
            diopiError_t ret = diopiadaptor::diopiGroupNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, mean, rstd, num_groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnique", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool return_counts, diopiTensorHandle_t indices, PtrWrapper<diopiTensor> counts) {
        if (diopiUnique) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiTensorHandle_t countsHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiUnique(ctx, &outHandle, input, dim, sorted, return_counts, indices, &countsHandle);
            if (out.get() != nullptr)
                 *out = *outHandle;
            if (counts.get() != nullptr)
                 *counts = *countsHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnique", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, bool sorted, bool return_counts, diopiTensorHandle_t indices, PtrWrapper<diopiTensor> counts) {
        if (diopiUnique) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiTensorHandle_t countsHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiUnique(ctx, &outHandle, input, nullptr, sorted, return_counts, indices, &countsHandle);
            if (out.get() != nullptr)
                 *out = *outHandle;
            if (counts.get() != nullptr)
                 *counts = *countsHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiProd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
        if (diopiProd) {
    
            diopiError_t ret = diopiadaptor::diopiProd(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiProd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiProd) {
    
            diopiError_t ret = diopiadaptor::diopiProd(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCTCLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
        if (diopiCTCLoss) {
    
            diopiError_t ret = diopiadaptor::diopiCTCLoss(ctx, out, neg_log_likelihood, log_alpha, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCTCLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
        if (diopiCTCLossBackward) {
    
            diopiError_t ret = diopiadaptor::diopiCTCLossBackward(ctx, grad_input, grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, reduction, zero_infinity);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLerpTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end, diopiConstTensorHandle_t weight) {
        if (diopiLerpTensor) {
    
            diopiError_t ret = diopiadaptor::diopiLerpTensor(ctx, out, input, end, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLerpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end, const diopiScalar_t* weight) {
        if (diopiLerpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiLerpScalar(ctx, out, input, end, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainderTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiRemainderTensor) {
    
            diopiError_t ret = diopiadaptor::diopiRemainderTensor(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainderScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiRemainderScalar) {
    
            diopiError_t ret = diopiadaptor::diopiRemainderScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainder", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
        if (diopiRemainder) {
    
            diopiError_t ret = diopiadaptor::diopiRemainder(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGather", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiGather) {
    
            diopiError_t ret = diopiadaptor::diopiGather(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGatherBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiGatherBackward) {
    
            diopiError_t ret = diopiadaptor::diopiGatherBackward(ctx, grad_input, grad_output, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterInp) {
    
            diopiError_t ret = diopiadaptor::diopiScatterInp(ctx, input, dim, src, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterInpScalar) {
    
            diopiError_t ret = diopiadaptor::diopiScatterInpScalar(ctx, input, dim, value, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatter) {
    
            diopiError_t ret = diopiadaptor::diopiScatter(ctx, out, input, dim, src, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterScalar) {
    
            diopiError_t ret = diopiadaptor::diopiScatterScalar(ctx, out, input, dim, value, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexPutInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        if (diopiIndexPutInp) {
            std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
            for (int i = 0; i < indices_counts; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiadaptor::diopiIndexPutInp(ctx, input, values, indicesDIOPI, indices_counts, accumulate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexPut", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        if (diopiIndexPut) {
            std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
            for (int i = 0; i < indices_counts; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiadaptor::diopiIndexPut(ctx, out, input, values, indicesDIOPI, indices_counts, accumulate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandomInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
        if (diopiRandomInp) {
    
            diopiError_t ret = diopiadaptor::diopiRandomInp(ctx, inout, from, to, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandomInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, int64_t idx) {
        if (diopiRandomInp) {
    
            diopiError_t ret = diopiadaptor::diopiRandomInp(ctx, inout, from, nullptr, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUniformInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
        if (diopiUniformInp) {
    
            diopiError_t ret = diopiadaptor::diopiUniformInp(ctx, inout, from, to, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulli", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
        if (diopiBernoulli) {
    
            diopiError_t ret = diopiadaptor::diopiBernoulli(ctx, out, input, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulliInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
        if (diopiBernoulliInp) {
    
            diopiError_t ret = diopiadaptor::diopiBernoulliInp(ctx, inout, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulliScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
        if (diopiBernoulliScalar) {
    
            diopiError_t ret = diopiadaptor::diopiBernoulliScalar(ctx, out, p, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArange", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
        if (diopiArange) {
    
            diopiError_t ret = diopiadaptor::diopiArange(ctx, out, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandperm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
        if (diopiRandperm) {
    
            diopiError_t ret = diopiadaptor::diopiRandperm(ctx, out, n, idx);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormal", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std) {
        if (diopiNormal) {
    
            diopiError_t ret = diopiadaptor::diopiNormal(ctx, out, mean, std);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalTensorScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std) {
        if (diopiNormalTensorScalar) {
    
            diopiError_t ret = diopiadaptor::diopiNormalTensorScalar(ctx, out, mean, std);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalScalarTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std) {
        if (diopiNormalScalarTensor) {
    
            diopiError_t ret = diopiadaptor::diopiNormalScalarTensor(ctx, out, mean, std);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std) {
        if (diopiNormalTensor) {
    
            diopiError_t ret = diopiadaptor::diopiNormalTensor(ctx, out, mean, std);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std) {
        if (diopiNormalInp) {
    
            diopiError_t ret = diopiadaptor::diopiNormalInp(ctx, inout, mean, std);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMeshGrid", [](diopiContextHandle_t ctx, py::list& outs, py::list& inputs, int64_t inputsNum) {
        if (diopiMeshGrid) {
            std::vector<diopiConstTensorHandle_t> inputsV(inputsNum);
            for (int i = 0; i < inputsNum; ++i)
                inputsV[i] = inputs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto inputsDIOPI = inputsV.data();
            std::vector<diopiTensorHandle_t> outsV(inputsNum);
            for (int i = 0; i < inputsNum; ++i)
                outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto outsDIOPI = outsV.data();
            diopiError_t ret = diopiadaptor::diopiMeshGrid(ctx, outsDIOPI, inputsDIOPI, inputsNum);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMultinomial", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {
        if (diopiMultinomial) {
    
            diopiError_t ret = diopiadaptor::diopiMultinomial(ctx, out, input, num_samples, replacement);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLayerNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape, double eps) {
        if (diopiLayerNorm) {
    
            diopiError_t ret = diopiadaptor::diopiLayerNorm(ctx, out, save_mean, save_invstd, input, weight, bias, normalized_shape, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLayerNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
        if (diopiLayerNormBackward) {
    
            diopiError_t ret = diopiadaptor::diopiLayerNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, mean, rstd, normalized_shape);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCopyInp", diopiCopyInp);
    m.def("diopiUpsampleNearest", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
        if (diopiUpsampleNearest) {
    
            diopiError_t ret = diopiadaptor::diopiUpsampleNearest(ctx, out, input, size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleNearestBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size) {
        if (diopiUpsampleNearestBackward) {
    
            diopiError_t ret = diopiadaptor::diopiUpsampleNearestBackward(ctx, grad_input, grad_output, out_size, in_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleLinear", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool align_corners, const char* mode) {
        if (diopiUpsampleLinear) {
    
            diopiError_t ret = diopiadaptor::diopiUpsampleLinear(ctx, out, input, size, align_corners, mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleLinearBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
        if (diopiUpsampleLinearBackward) {
    
            diopiError_t ret = diopiadaptor::diopiUpsampleLinearBackward(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfinv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiErfinv) {
    
            diopiError_t ret = diopiadaptor::diopiErfinv(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfinvInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiErfinvInp) {
    
            diopiError_t ret = diopiadaptor::diopiErfinvInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIm2Col", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
        if (diopiIm2Col) {
    
            diopiError_t ret = diopiadaptor::diopiIm2Col(ctx, out, input, kernel_size, dilation, padding, stride);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCol2Im", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
        if (diopiCol2Im) {
    
            diopiError_t ret = diopiadaptor::diopiCol2Im(ctx, out, input, output_size, kernel_size, dilation, padding, stride);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRepeat", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
        if (diopiRepeat) {
    
            diopiError_t ret = diopiadaptor::diopiRepeat(ctx, out, input, repeats_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCastDtype", diopiCastDtype);
    m.def("diopiPolar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle) {
        if (diopiPolar) {
    
            diopiError_t ret = diopiadaptor::diopiPolar(ctx, out, abs, angle);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
        if (diopiTriu) {
    
            diopiError_t ret = diopiadaptor::diopiTriu(ctx, out, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriuInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
        if (diopiTriuInp) {
    
            diopiError_t ret = diopiadaptor::diopiTriuInp(ctx, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgn", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSgn) {
    
            diopiError_t ret = diopiadaptor::diopiSgn(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgnInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSgnInp) {
    
            diopiError_t ret = diopiadaptor::diopiSgnInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIsNan", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiIsNan) {
    
            diopiError_t ret = diopiadaptor::diopiIsNan(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinalgQR", [](diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char* mode, diopiTensorHandle_t Q, diopiTensorHandle_t R) {
        if (diopiLinalgQR) {
    
            diopiError_t ret = diopiadaptor::diopiLinalgQR(ctx, A, mode, Q, R);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t self, diopiSize_t dim, bool keepdim) {
        if (diopiAmax) {
    
            diopiError_t ret = diopiadaptor::diopiAmax(ctx, out, self, dim, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiContiguous", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
        if (diopiContiguous) {
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiadaptor::diopiContiguous(ctx, &outHandle, input, memoryFormat);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
}
// NOLINTEND
