/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

//NOLINTBEGIN
#include <pybind11/pybind11.h>
#include "litert.hpp"
#include <diopi/diopirt.h>
#include <diopi/functions.h>

namespace py = pybind11;

PYBIND11_MODULE(export_functions, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring
    m.def("diopiGetVendorName", &diopiGetVendorName);
    m.def("diopiGetImplVersion", &diopiGetImplVersion);
    m.def("diopiGetVersion", &diopiGetVersion);
    m.def("diopiGetLastErrorString", &diopiGetLastErrorString);
    m.def("diopiConvolution2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvolution2d(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvolution2dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
        if (diopiBatchNorm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNorm(ctx, out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormStats", [](diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, double eps) {
        if (diopiBatchNormStats) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormStats(ctx, mean, invstd, input, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormGatherStatsWithCounts", [](diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean_all, diopiConstTensorHandle_t invstd_all, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, float momentum, float eps, diopiConstTensorHandle_t counts) {
        if (diopiBatchNormGatherStatsWithCounts) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormGatherStatsWithCounts(ctx, mean, invstd, input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackwardReduce", [](diopiContextHandle_t ctx, diopiTensorHandle_t sum_dy, diopiTensorHandle_t sum_dy_xmu, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, bool input_g, bool weight_g, bool bias_g) {
        if (diopiBatchNormBackwardReduce) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormBackwardReduce(ctx, sum_dy, sum_dy_xmu, grad_weight, grad_bias, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackwardElemt", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t sum_dy, diopiConstTensorHandle_t sum_dy_xmu, diopiConstTensorHandle_t count) {
        if (diopiBatchNormBackwardElemt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormBackwardElemt(ctx, grad_input, grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormElemt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps) {
        if (diopiBatchNormElemt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormElemt(ctx, out, input, weight, bias, mean, invstd, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBatchNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
        if (diopiBatchNormBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBatchNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, training, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiRelu) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRelu(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiReluInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiReluInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanh", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanh) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardtanh(ctx, out, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanhInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanhInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardtanhInp(ctx, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardtanhBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
        if (diopiHardtanhBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardtanhBackward(ctx, grad_input, grad_output, input, min_val, max_val);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswish", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiHardswish) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardswish(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswishInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiHardswishInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardswishInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiHardswishBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiHardswishBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiHardswishBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThreshold", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
        if (diopiThreshold) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiThreshold(ctx, out, input, threshold, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThresholdInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
        if (diopiThresholdInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiThresholdInp(ctx, input, threshold, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiThresholdBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
        if (diopiThresholdBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiThresholdBackward(ctx, grad_input, grad_output, input, threshold);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
        if (diopiGelu) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGelu(ctx, out, input, approximate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const char* approximate) {
        if (diopiGeluBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGeluBackward(ctx, grad_input, grad_output, input, approximate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyRelu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
        if (diopiLeakyRelu) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeakyRelu(ctx, out, input, negative_slope);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyReluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
        if (diopiLeakyReluInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeakyReluInp(ctx, input, negative_slope);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeakyReluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
        if (diopiLeakyReluBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeakyReluBackward(ctx, grad_input, grad_output, input, negative_slope, input_is_result);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
        if (diopiAvgPool2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAvgPool2d(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
        if (diopiAvgPool2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAvgPool2d(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
        if (diopiAvgPool2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAvgPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
        if (diopiAvgPool2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAvgPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool2d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool2dWithIndices) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool2dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
        if (diopiMaxPool2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveAvgPool2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveAvgPool2d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiAdaptiveAvgPool2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveAvgPool2dBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool2d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool2dWithIndices) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool2dWithIndices(ctx, out, indices, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
        if (diopiAdaptiveMaxPool2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool2dBackward(ctx, grad_input, grad_output, input, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDropout", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train, diopiGeneratorHandle_t generator) {
        if (diopiDropout) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDropout(ctx, out, mask, input, p, train, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDropoutInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train, diopiGeneratorHandle_t generator) {
        if (diopiDropoutInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDropoutInp(ctx, input, mask, p, train, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMSELoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
        if (diopiMSELoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMSELoss(ctx, out, input, target, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMSELossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
        if (diopiMSELossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMSELossBackward(ctx, grad_input, grad_output, input, target, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidFocalLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
        if (diopiSigmoidFocalLoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSigmoidFocalLoss(ctx, out, inputs, targets, alpha, gamma, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidFocalLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
        if (diopiSigmoidFocalLossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSigmoidFocalLossBackward(ctx, grad_output, input, target, grad_input, gamma, alpha, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCrossEntropyLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
        if (diopiCrossEntropyLoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCrossEntropyLoss(ctx, out, input, target, weight, reduction, ignore_index, label_smoothing);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCrossEntropyLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
        if (diopiCrossEntropyLossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCrossEntropyLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index, label_smoothing);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNLLLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
        if (diopiNLLLoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNLLLoss(ctx, out, input, target, weight, reduction, ignore_index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNLLLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
        if (diopiNLLLossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNLLLossBackward(ctx, grad_input, grad_output, input, target, weight, reduction, ignore_index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCEWithLogits", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
        if (diopiBCEWithLogits) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBCEWithLogits(ctx, out, input, target, weight, pos_weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCEWithLogitsBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
        if (diopiBCEWithLogitsBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBCEWithLogitsBackward(ctx, grad_input, grad_output, input, target, weight, pos_weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCELoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
        if (diopiBCELoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBCELoss(ctx, out, input, target, weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBCELossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
        if (diopiBCELossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBCELossBackward(ctx, grad_input, grad_output, input, target, weight, reduction);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSign", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSign) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSign(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAbsInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAbsInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAbsInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAbs", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAbs) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAbs(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNegInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiNegInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNegInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeg", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiNeg) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNeg(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFloorInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiFloorInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiFloorInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFloor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiFloor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiFloor(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCeilInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiCeilInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCeilInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCeil", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiCeil) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCeil(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSqrtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSqrtInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSqrtInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSqrt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSqrt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSqrt(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRsqrtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiRsqrtInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRsqrtInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRsqrt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiRsqrt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRsqrt(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSinInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSinInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSin) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSin(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAsinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAsinInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAsinInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAsin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAsin) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAsin(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCosInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiCosInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCosInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCos", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiCos) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCos(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanhInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiTanhInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTanhInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanh", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiTanh) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTanh(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTanhBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
        if (diopiTanhBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTanhBackward(ctx, grad_input, grad_output, output);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAtan", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAtan) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAtan(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAtanInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiAtanInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAtanInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSigmoidInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSigmoidInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoid", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSigmoid) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSigmoid(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSigmoidBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
        if (diopiSigmoidBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSigmoidBackward(ctx, grad_input, grad_output, output);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSiluInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSiluInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSiluInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSilu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSilu) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSilu(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSiluBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiSiluBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSiluBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExpInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiExpInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiExpInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExp", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiExp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiExp(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLogInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLog(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog2Inp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLog2Inp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLog2Inp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog2", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog2) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLog2(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog10Inp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLog10Inp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLog10Inp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLog10", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLog10) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLog10(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiErfInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiErfInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErf", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiErf) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiErf(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
        if (diopiPowScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPowScalar(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPow", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
        if (diopiPow) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPow(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
        if (diopiPowInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPowInp(ctx, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
        if (diopiPowTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPowTensor(ctx, out, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPowInpTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
        if (diopiPowInpTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPowInpTensor(ctx, input, exponent);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiAdd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdd(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiAddInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddInp(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiAddScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddScalar(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiAddInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddInpScalar(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSub", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiSub) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSub(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
        if (diopiSubInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSubInp(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiSubScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSubScalar(ctx, out, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSubInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
        if (diopiSubInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSubInpScalar(ctx, input, other, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMul) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMul(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMulInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMulInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiMulScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMulScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMulInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiMulInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMulInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDiv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
        if (diopiDiv) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDiv(ctx, out, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
        if (diopiDivInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDivInp(ctx, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
        if (diopiDivScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDivScalar(ctx, out, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiDivInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
        if (diopiDivInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiDivInpScalar(ctx, input, other, rounding_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
        if (diopiBmm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBmm(ctx, out, input, mat2);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBaddbmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
        if (diopiBaddbmm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBaddbmm(ctx, out, input, batch1, batch2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBaddbmmInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
        if (diopiBaddbmmInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBaddbmmInp(ctx, input, batch1, batch2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcmul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcmul) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddcmul(ctx, out, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcmulInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcmulInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddcmulInp(ctx, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMatmul", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMatmul) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMatmul(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcdiv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcdiv) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddcdiv(ctx, out, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddcdivInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
        if (diopiAddcdivInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddcdivInp(ctx, input, tensor1, tensor2, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAddmm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
        if (diopiAddmm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAddmm(ctx, out, input, mat1, mat2, beta, alpha);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCholesky", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror) {
        if (diopiCholesky) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCholesky(ctx, out, info, mat, upper, checkerror);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCholeskyBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper) {
        if (diopiCholeskyBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCholeskyBackward(ctx, grad_mat, grad_output, L, upper);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriangularSolve", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
        if (diopiTriangularSolve) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTriangularSolve(ctx, out, cloned_mat, b, mat, upper, transpose, unitriangular);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriangularSolveBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
        if (diopiTriangularSolveBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTriangularSolveBackward(ctx, grad_b, grad_mat, grad_x, grad_cloned_mat, x, b, mat, upper, transpose, unitriangular);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
        if (diopiClampInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampInpScalar(ctx, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
        if (diopiClampInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampInp(ctx, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
        if (diopiClampScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampScalar(ctx, out, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClamp", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
        if (diopiClamp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClamp(ctx, out, input, min, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
        if (diopiClampMaxInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMaxInpScalar(ctx, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
        if (diopiClampMaxInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMaxInp(ctx, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMaxScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
        if (diopiClampMaxScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMaxScalar(ctx, out, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
        if (diopiClampMax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMax(ctx, out, input, max);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
        if (diopiClampMinInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMinInpScalar(ctx, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
        if (diopiClampMinInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMinInp(ctx, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMinScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
        if (diopiClampMinScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMinScalar(ctx, out, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClampMin", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
        if (diopiClampMin) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiClampMin(ctx, out, input, min);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
        if (diopiFill) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiFill(ctx, input, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalAnd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalAnd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalAnd(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalAndInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalAndInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalAndInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalOr", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalOr) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalOr(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalOrInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLogicalOrInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalOrInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalNot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiLogicalNot) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalNot(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogicalNotInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiLogicalNotInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogicalNotInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAnd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseAnd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseAnd(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseAndInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseAndInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseAndScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseAndScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseAndInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseAndInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseAndInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOr", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseOr) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseOr(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiBitwiseOrInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseOrInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseOrScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseOrScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseOrInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiBitwiseOrInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseOrInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseNot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiBitwiseNot) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseNot(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBitwiseNotInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiBitwiseNotInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBitwiseNotInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiEqScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEqScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiEqInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEqInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEq", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiEq) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEq(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEqInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiEqInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEqInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiNeScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiNeInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiNe) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiNeInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGeScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGeInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGe) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGeInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGtScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGtScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiGtInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGtInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGt(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiGtInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGtInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLeScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLeInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLe", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLe) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLe(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLeInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLeInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLeInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLtScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLtScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiLtInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLtInpScalar(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLt", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLt) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLt(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLtInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiLtInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLtInp(ctx, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMean", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
        if (diopiMean) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMean(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
        if (diopiSum) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSum(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiStd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
        if (diopiStd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiStd(ctx, out, input, dim, unbiased);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMin", [](diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiMin) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMin(ctx, min, min_indices, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMinAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
        if (diopiMinAll) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMinAll(ctx, min, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMax", [](diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiMax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMax(ctx, max, max_indices, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
        if (diopiMaxAll) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxAll(ctx, max, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAny", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
        if (diopiAny) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAny(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAny", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAny) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAny(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
        if (diopiAll) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAll(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiAll) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAll(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSoftmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiSoftmax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSoftmax(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSoftmaxBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
        if (diopiSoftmaxBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSoftmaxBackward(ctx, grad_input, grad_output, output, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogSoftmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiLogSoftmax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogSoftmax(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLogSoftmaxBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
        if (diopiLogSoftmaxBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLogSoftmaxBackward(ctx, grad_input, grad_output, output, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndex", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, py::list& indices, int64_t nums) {
        if (diopiIndex) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> indicesV(nums);
            for (int i = 0; i < nums; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiIndex(ctx, &outHandle, input, indicesDIOPI, nums);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input, py::list& indices, int64_t nums, diopiConstTensorHandle_t grad_output) {
        if (diopiIndexBackward) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> indicesV(nums);
            for (int i = 0; i < nums; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiIndexBackward(ctx, grad_input, zeros_like_input, indicesDIOPI, nums, grad_output);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexSelect", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiIndexSelect) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexSelect(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiIndexSelectBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexSelectBackward(ctx, grad_input, grad, input_sizes, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelect", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
        if (diopiSelect) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSelect(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
        if (diopiSelectBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSelectBackward(ctx, grad_input, grad_output, input_sizes, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSelectScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t index) {
        if (diopiSelectScatter) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSelectScatter(ctx, out, input, src, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSliceScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSliceScatter) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSliceScatter(ctx, out, input, src, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSlice", [](diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSlice) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSlice(ctx, null_out, input, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSliceBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
        if (diopiSliceBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSliceBackward(ctx, grad_input, grad_output, input_sizes, dim, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
        if (diopiMaskedScatter) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedScatter(ctx, out, input, mask, source);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNms", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t confidence, double iou_threshold) {
        if (diopiNms) {
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiNms(ctx, &outHandle, boxes, confidence, iou_threshold);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNonzero", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input) {
        if (diopiNonzero) {
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiNonzero(ctx, &outHandle, input);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinear", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
        if (diopiLinear) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLinear(ctx, out, input, weight, bias);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinearBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
        if (diopiLinearBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLinearBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoiAlign", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
        if (diopiRoiAlign) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRoiAlign(ctx, out, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoiAlignBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height, int64_t width, int64_t sampling_ratio, bool aligned) {
        if (diopiRoiAlignBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRoiAlignBackward(ctx, out, grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgd", [](diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
        if (diopiSgd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSgd(ctx, w, dw, buf, lr, momentum, dampening, weight_decay, nesterov);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiClipGradNorm", [](diopiContextHandle_t ctx, void* out, py::list& grads, int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {
        if (diopiClipGradNorm) {
            py::gil_scoped_release no_gil;
            std::vector<diopiTensorHandle_t> gradsV(num_grads);
            for (int i = 0; i < num_grads; ++i)
                gradsV[i] = grads[i].cast<PtrWrapper<diopiTensor>>().get();
            auto gradsDIOPI = gradsV.data();
            diopiError_t ret = diopiClipGradNorm(ctx, reinterpret_cast<double*>(out), gradsDIOPI, num_grads, max_norm, norm_type, error_if_nonfinite);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbeddingRenorm_", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
        if (diopiEmbeddingRenorm_) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEmbeddingRenorm_(ctx, inout, indices, max_norm, norm_type);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbedding", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
        if (diopiEmbedding) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEmbedding(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiEmbeddingBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
        if (diopiEmbeddingBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiEmbeddingBackward(ctx, out, grad, indices, num_weights, padding_idx, scale_grad_byfreq, sparse);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTril", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
        if (diopiTril) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTril(ctx, out, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCat", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t num_inputs, int64_t dim) {
        if (diopiCat) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> tensorsV(num_inputs);
            for (int i = 0; i < num_inputs; ++i)
                tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
            auto tensorsDIOPI = tensorsV.data();
            diopiError_t ret = diopiCat(ctx, out, tensorsDIOPI, num_inputs, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSplitWithSizes", [](diopiContextHandle_t ctx, py::list& outs, int64_t num_outs, diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
        if (diopiSplitWithSizes) {
            py::gil_scoped_release no_gil;
            std::vector<diopiTensorHandle_t> outsV(num_outs);
            for (int i = 0; i < num_outs; ++i)
                outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto outsDIOPI = outsV.data();
            diopiError_t ret = diopiSplitWithSizes(ctx, outsDIOPI, num_outs, input, splitSizes, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiStack", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t numTensors, int64_t dim) {
        if (diopiStack) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> tensorsV(numTensors);
            for (int i = 0; i < numTensors; ++i)
                tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
            auto tensorsDIOPI = tensorsV.data();
            diopiError_t ret = diopiStack(ctx, out, tensorsDIOPI, numTensors, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSort", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
        if (diopiSort) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSort(ctx, values, indices, input, dim, descending, stable);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSort", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending) {
        if (diopiSort) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSort(ctx, values, indices, input, dim, descending, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTopk", [](diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
        if (diopiTopk) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTopk(ctx, values, indices, input, k, dim, largest, sorted);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTranspose", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
        if (diopiTranspose) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTranspose(ctx, out, input, dim0, dim1);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiOneHot", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes) {
        if (diopiOneHot) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiOneHot(ctx, out, input, num_classes);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiWhere", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiWhere) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiWhere(ctx, out, condition, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
        if (diopiMaskedFill) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedFill(ctx, out, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
        if (diopiMaskedFillInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedFillInp(ctx, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
        if (diopiMaskedFillScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedFillScalar(ctx, out, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedFillInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
        if (diopiMaskedFillInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedFillInpScalar(ctx, input, mask, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReciprocal", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiReciprocal) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiReciprocal(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiReciprocalInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiReciprocalInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiReciprocalInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdamW", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
        if (diopiAdamW) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdamW(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvTranspose2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
        if (diopiConvTranspose2d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvTranspose2d(ctx, out, input, weight, bias, stride, padding, output_padding, groups, dilation);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvTranspose2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, diopiSize_t output_padding, int64_t groups) {
        if (diopiConvTranspose2dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvTranspose2dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, output_padding, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnfold", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
        if (diopiUnfold) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUnfold(ctx, out, input, dim, size, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnfoldBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
        if (diopiUnfoldBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUnfoldBackward(ctx, grad_input, grad_output, input_sizes, dim, size, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCumsum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
        if (diopiCumsum) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCumsum(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdist", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, const int64_t* compute_mode) {
        if (diopiCdist) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCdist(ctx, out, input1, input2, p, compute_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdist", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p) {
        if (diopiCdist) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCdist(ctx, out, input1, input2, p, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCdistBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
        if (diopiCdistBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCdistBackward(ctx, grad_input, grad_output, input1, input2, p, cdist);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArgmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
        if (diopiArgmax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiArgmax(ctx, out, input, dim, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArgmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, bool keepdim) {
        if (diopiArgmax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiArgmax(ctx, out, input, nullptr, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdadelta", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
        if (diopiAdadelta) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdadelta(ctx, input, grad, square_avg, acc_delta, lr, rho, eps, weight_decay);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdam", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
        if (diopiAdam) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdam(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRmsprop", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered) {
        if (diopiRmsprop) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRmsprop(ctx, input, grad, square_avg, grad_avg, momentum_buf, lr, alpha, eps, weight_decay, momentum, centered);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSmoothL1Loss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
        if (diopiSmoothL1Loss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSmoothL1Loss(ctx, out, input, target, reduction, beta);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSmoothL1LossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
        if (diopiSmoothL1LossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSmoothL1LossBackward(ctx, grad_input, grad_output, input, target, reduction, beta);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution3d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvolution3d(ctx, out, input, weight, bias, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiConvolution3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
        if (diopiConvolution3dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiConvolution3dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool3d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool3d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
        if (diopiMaxPool3dWithIndices) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool3dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaxPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
        if (diopiMaxPool3dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaxPool3dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveAvgPool3d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveAvgPool3d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveAvgPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
        if (diopiAdaptiveAvgPool3dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveAvgPool3dBackward(ctx, grad_input, grad_output, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool3d) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool3d(ctx, out, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3dWithIndices", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
        if (diopiAdaptiveMaxPool3dWithIndices) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool3dWithIndices(ctx, out, indices, input, output_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAdaptiveMaxPool3dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
        if (diopiAdaptiveMaxPool3dBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAdaptiveMaxPool3dBackward(ctx, grad_input, grad_output, input, indices);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedSelect", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
        if (diopiMaskedSelect) {
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiMaskedSelect(ctx, &outHandle, input, mask);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaskedSelectBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
        if (diopiMaskedSelectBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaskedSelectBackward(ctx, grad_input, grad_output, input, mask);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMaximum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMaximum) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMaximum(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMinimum", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiMinimum) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMinimum(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
        if (diopiMm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMm(ctx, out, input, mat2);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
        if (diopiIndexFillScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexFillScalar(ctx, out, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFill", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
        if (diopiIndexFill) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexFill(ctx, out, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
        if (diopiIndexFillInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexFillInpScalar(ctx, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexFillInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
        if (diopiIndexFillInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIndexFillInp(ctx, input, dim, index, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiExpand", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiExpand) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiExpand(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinspace", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
        if (diopiLinspace) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLinspace(ctx, out, start, end, steps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPermute", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
        if (diopiPermute) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPermute(ctx, out, input, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPad", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, const double* value) {
        if (diopiPad) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPad(ctx, out, input, pad, mode, value);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiPad", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode) {
        if (diopiPad) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPad(ctx, out, input, pad, mode, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRoll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
        if (diopiRoll) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRoll(ctx, out, input, shifts, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiFlip", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
        if (diopiFlip) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiFlip(ctx, out, input, dims);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
        if (diopiNorm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNorm(ctx, out, input, p, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGroupNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
        if (diopiGroupNorm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGroupNorm(ctx, out, save_mean, save_invstd, input, weight, bias, num_groups, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGroupNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t num_groups) {
        if (diopiGroupNormBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGroupNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, mean, rstd, num_groups);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUnique", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool return_counts, diopiTensorHandle_t indices, PtrWrapper<diopiTensor> counts) {
        if (diopiUnique) {
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiTensorHandle_t countsHandle = nullptr;
            diopiError_t ret = diopiUnique(ctx, &outHandle, input, dim, sorted, return_counts, indices, &countsHandle);
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
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiTensorHandle_t countsHandle = nullptr;
            diopiError_t ret = diopiUnique(ctx, &outHandle, input, nullptr, sorted, return_counts, indices, &countsHandle);
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
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiProd(ctx, out, input, dim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiProd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiProd) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiProd(ctx, out, input, nullptr);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCTCLoss", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
        if (diopiCTCLoss) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCTCLoss(ctx, out, neg_log_likelihood, log_alpha, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCTCLossBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
        if (diopiCTCLossBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCTCLossBackward(ctx, grad_input, grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, reduction, zero_infinity);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLerpTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end, diopiConstTensorHandle_t weight) {
        if (diopiLerpTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLerpTensor(ctx, out, input, end, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLerpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end, const diopiScalar_t* weight) {
        if (diopiLerpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLerpScalar(ctx, out, input, end, weight);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainderTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
        if (diopiRemainderTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRemainderTensor(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainderScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
        if (diopiRemainderScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRemainderScalar(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRemainder", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
        if (diopiRemainder) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRemainder(ctx, out, input, other);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGather", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiGather) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGather(ctx, out, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiGatherBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
        if (diopiGatherBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiGatherBackward(ctx, grad_input, grad_output, input, dim, index);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiScatterInp(ctx, input, dim, src, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterInpScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterInpScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiScatterInpScalar(ctx, input, dim, value, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatter", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatter) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiScatter(ctx, out, input, dim, src, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiScatterScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
        if (diopiScatterScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiScatterScalar(ctx, out, input, dim, value, index, reduce);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexPutInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        if (diopiIndexPutInp) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
            for (int i = 0; i < indices_counts; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiIndexPutInp(ctx, input, values, indicesDIOPI, indices_counts, accumulate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIndexPut", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        if (diopiIndexPut) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
            for (int i = 0; i < indices_counts; ++i)
                indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
            auto indicesDIOPI = indicesV.data();
            diopiError_t ret = diopiIndexPut(ctx, out, input, values, indicesDIOPI, indices_counts, accumulate);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandomInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, diopiGeneratorHandle_t generator) {
        if (diopiRandomInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRandomInp(ctx, inout, from, to, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandomInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, diopiGeneratorHandle_t generator) {
        if (diopiRandomInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRandomInp(ctx, inout, from, nullptr, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUniformInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
        if (diopiUniformInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUniformInp(ctx, inout, from, to, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulli", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiGeneratorHandle_t generator) {
        if (diopiBernoulli) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBernoulli(ctx, out, input, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulliInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiGeneratorHandle_t generator) {
        if (diopiBernoulliInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBernoulliInp(ctx, inout, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiBernoulliScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
        if (diopiBernoulliScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiBernoulliScalar(ctx, out, p, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiArange", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
        if (diopiArange) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiArange(ctx, out, start, end, step);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRandperm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator) {
        if (diopiRandperm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRandperm(ctx, out, n, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormal", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
        if (diopiNormal) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNormal(ctx, out, mean, std, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalTensorScalar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std, diopiGeneratorHandle_t generator) {
        if (diopiNormalTensorScalar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNormalTensorScalar(ctx, out, mean, std, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalScalarTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std, diopiGeneratorHandle_t generator) {
        if (diopiNormalScalarTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNormalScalarTensor(ctx, out, mean, std, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalTensor", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std, diopiGeneratorHandle_t generator) {
        if (diopiNormalTensor) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNormalTensor(ctx, out, mean, std, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiNormalInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
        if (diopiNormalInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiNormalInp(ctx, inout, mean, std, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMeshGrid", [](diopiContextHandle_t ctx, py::list& outs, py::list& inputs, int64_t inputsNum) {
        if (diopiMeshGrid) {
            py::gil_scoped_release no_gil;
            std::vector<diopiConstTensorHandle_t> inputsV(inputsNum);
            for (int i = 0; i < inputsNum; ++i)
                inputsV[i] = inputs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto inputsDIOPI = inputsV.data();
            std::vector<diopiTensorHandle_t> outsV(inputsNum);
            for (int i = 0; i < inputsNum; ++i)
                outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
            auto outsDIOPI = outsV.data();
            diopiError_t ret = diopiMeshGrid(ctx, outsDIOPI, inputsDIOPI, inputsNum);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiMultinomial", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement, diopiGeneratorHandle_t generator) {
        if (diopiMultinomial) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiMultinomial(ctx, out, input, num_samples, replacement, generator);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLayerNorm", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape, double eps) {
        if (diopiLayerNorm) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLayerNorm(ctx, out, save_mean, save_invstd, input, weight, bias, normalized_shape, eps);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLayerNormBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
        if (diopiLayerNormBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLayerNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias, mean, rstd, normalized_shape);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCopyInp", diopiCopyInp);
    m.def("diopiUpsampleNearest", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
        if (diopiUpsampleNearest) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUpsampleNearest(ctx, out, input, size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleNearestBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size) {
        if (diopiUpsampleNearestBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUpsampleNearestBackward(ctx, grad_input, grad_output, out_size, in_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleLinear", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool align_corners, const char* mode) {
        if (diopiUpsampleLinear) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUpsampleLinear(ctx, out, input, size, align_corners, mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiUpsampleLinearBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
        if (diopiUpsampleLinearBackward) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiUpsampleLinearBackward(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfinv", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiErfinv) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiErfinv(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiErfinvInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiErfinvInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiErfinvInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIm2Col", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
        if (diopiIm2Col) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIm2Col(ctx, out, input, kernel_size, dilation, padding, stride);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCol2Im", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
        if (diopiCol2Im) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiCol2Im(ctx, out, input, output_size, kernel_size, dilation, padding, stride);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiRepeat", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
        if (diopiRepeat) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiRepeat(ctx, out, input, repeats_size);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiCastDtype", diopiCastDtype);
    m.def("diopiPolar", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle) {
        if (diopiPolar) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiPolar(ctx, out, abs, angle);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriu", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
        if (diopiTriu) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTriu(ctx, out, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiTriuInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
        if (diopiTriuInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiTriuInp(ctx, input, diagonal);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgn", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiSgn) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSgn(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiSgnInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input) {
        if (diopiSgnInp) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiSgnInp(ctx, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiIsNan", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
        if (diopiIsNan) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiIsNan(ctx, out, input);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiLinalgQR", [](diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char* mode, diopiTensorHandle_t Q, diopiTensorHandle_t R) {
        if (diopiLinalgQR) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiLinalgQR(ctx, A, mode, Q, R);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiAmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t self, diopiSize_t dim, bool keepdim) {
        if (diopiAmax) {
            py::gil_scoped_release no_gil;
    
            diopiError_t ret = diopiAmax(ctx, out, self, dim, keepdim);
    
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
    m.def("diopiContiguous", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
        if (diopiContiguous) {
            py::gil_scoped_release no_gil;
            diopiTensorHandle_t outHandle = nullptr;
            diopiError_t ret = diopiContiguous(ctx, &outHandle, input, memoryFormat);
            if (out.get() != nullptr)
                 *out = *outHandle;
            return ret;
        } else {
            return diopiError_t::diopiNoImplement;
        }
    });
}
// NOLINTEND
