/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

typedef enum { ReductionNone, ReductionMean, ReductionSum, ReductionEND } diopiReduction_t;

typedef enum { RoundModeNone, RoundModeTrunc, RoundModeFloor, RoundModeEND } diopiRoundMode_t;

typedef struct {
    diopiDtype_t stype;
    union {
        double fval;
        int64_t ival;
    };
    diopiDtype_t type() { return stype; }
    double val() {
        if (stype == diopiDtype_t::diopi_dtype_float64)
            return fval;
        else if (stype == diopiDtype_t::diopi_dtype_int64)
            return ival;
    }
} diopiScalar_t;

typedef enum { Contiguous = 0, ChannelsLast = 1, ChannelsLast3d = 2, Preserve = 3 } diopiMemoryFormat_t;

/**
 * \brief get the vendor's name who implements the functions
 */
DIOPI_RT_API const char* diopiGetVendorName();
DIOPI_RT_API const char* diopiGetImplVersion();
DIOPI_RT_API const char* diopiGetLastErrorString();

/**
 * @brief Applies a 2D convolution over an input image composed of several input planes.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float16, float64].
 * @param weight the weight tensor; dimension of kernel_size must match the number of input spatial dimensions.
 * type = [float32, float16, float64].
 * @param bias bias tensor. type = [float32, float16, float64].
 * @param stride an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param padding an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param dilation an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param groups number of groups for grouped convolution. type = [int64].
 * @param[out] out the result tensor. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups);

/**
 * @brief Backward pass for convolution2d. Computes gradients for input, weight, and bias.
 * @param[in] grad_output the grad tensor of output. type = [float32, float16, float64].
 * @param bias_sizes an array, indicates that a bias was used in the forward pass and contains the shape of the bias. type = [int32, int64].
 * @param transposed indicating whether the convolution is transposed.
 * @param output_padding an array, dimension == number of input spatial dimensions; only supported when transposed is true. type = [int32, int64].
 * @param[out] grad_input the grad of input. type = [float32, float16, float64].
 * @param grad_weight the grad of weight. type = [float32, float16, float64].
 * @param grad_bias the grad of bias. type = [float32, float16, float64].
 * @sa Other parameters refer to diopiConvolution2d().
 */
DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups);

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
DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps);

/**
 * @brief compute the backward pass of batch normalization
 * @param[in] grad_output Gradient of normalized layer output, with the same shape as the forward pass output. type=[float32, float16, float64].
 * @param[out] grad_input Gradient of the input data, with the same shape as the input data. type = [float32, float16, float64].
 * @param grad_weight Gradient of the weight parameter, with the same shape as the weight parameter. type = [float32, float16, float64].
 * @param grad_bias Gradient of the bias parameter, with the same shape as the bias parameter. type = [float32, float16, float64].
 * @sa Other parameters refer to diopiBatchNorm().
 */
DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var,
                                              diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps);

/**
 * @brief Applies the rectified linear unit function element-wise.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64].
 * @param[out] out the result tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiRelu().
 * @param[in] ctx Context environment.
 * @param input the input tensor and will be stored result tensor.type = [float32, float64].
 */
DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief It clips the tensor values within a range defined by the lower and upper bounds.
 * Any values below the lower bound are set to the lower bound, and any values above the upper bound are set to the upper bound.
 * @param[in] ctx Context environment.
 * @param input the input tensor,type = [float32, float64].
 * @param min_val scalar, the lower bound. type = [int, float].
 * @param max_val scalar, the upper bound. type = [int, float].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val,
                                     const diopiScalar_t* max_val);
/**
 * @brief the in-place version of diopiHardtanh().
 * @param input the input tensor and will be stored result tensor. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);

/**
 * @brief compute the backward pass of diopiHardtanhInp().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);
/**
 * @brief Applies the Hardswish function, element-wise, as described in the paper:`Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.
 *    Hardswish is defined as:
    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}
 * @param[in] ctx Context environment.
 * @param input the input tensor,type = [float16, float32, float64].
 * @param out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief the in-place version of diopiHardtanh().
 * @param input the input tensor and will be stored result tensor. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiHardswish().
 */
DIOPI_API diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief compute the backward pass of diopiHardswish().
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiHardswishInp().
 */
DIOPI_API diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input);
/**
 * @brief The function thresholds the input tensor by setting elements greater than a given threshold to the threshold value, while leaving elements less than
 * or equal to the threshold unchanged.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64].
 * @param threshold the value to threshold at. type = [int, float].
 * @param value the value to replace with. type = [int, float].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                                      const diopiScalar_t* value);

/**
 * @brief the in-place version of diopiThreshold().
 * @param[in] ctx Context environment.
 * @param input the input tensor and will be stored result tensor. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiThreshold().
 */
DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value);

/**
 * @brief compute the backward pass of diopiThreshold().
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiThreshold().
 */
DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold);

/**
 * @brief Applies the gaussian error linear unit function element-wise
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64].
 * @param approximate Whether to use an approximate estimation. If it equals to "tanh", it will use an approximate estimation.
 * @param[out] out theout put tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate);
/**
 * @brief compute the backward pass of diopiGelu().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char* approximate);

/**
 * @brief Applies element-wise, LeakyReLU(x) = max(0,x) + negative_slope*min(0,x)
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64].
 * @param negative_slope Controls the angle of the negative slope. type = [int, float].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope);
/**
 * @brief the in-place version of diopiLeakyRelu().
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float32, float64].
 * @sa Other parameters refer to diopiLeakyRelu().
 */
DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope);
/**
 * @brief compute the backward pass of diopiLeakyRelu().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param input_is_result boolean.
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiLeakyRelu().
 */
DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result);

/**
 * @brief Applies 2D average-pooling operation in kH×kW regions by step size sH×sW steps.
 * @param[in] ctx Context environment.
 * @param input input tensor, type = [float32, float64]
 * @param kernel_size an array, the size of the pooling region. type = [int32, int64].
 * @param stride an array, the stride of the pooling operation. type = [int32, int64].
 * @param padding an array. type = [int32, int64].
 * @param ceil_mode boolean, when set to True, uses ceil instead of floor in the formula to compute the output shape.
 * @param count_include_pad boolean, when True, zero-padding will be included in the mean calculation.
 * @param divisor_override If specified, it will be used as the divisor when computing the average pooling,
 *  otherwise the default is to divide by the total number of pooling elements.
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override);

/**
 * @brief compute the backward pass of diopiAvgPool2d().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiAvgPool2d().
 */
DIOPI_API diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                              bool count_include_pad, const int64_t* divisor_override);

/**
 * @brief Applies a 2D max pooling over an input signal composed of several input planes
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32]
 * @param kernel_size an array, size of the pooling region. type = [int32, int64].
 * @param stride an array, stride of the pooling operation. type = [int32, int64].
 * @param padding  an array, implicit negative infinity padding on both sides of the input tensor, its value should be >= 0 and <= kernel_size / 2. type =
 * [int32, int64].
 * @param dilation an array, spacing between the elements within the sliding window, its value should be greater than 0. type = [int32, int64].
 * @param ceil_mode boolean, if True, use ceil instead of the default floor operation when computing the output shape.
 * This ensures that every element in the input tensor is covered by a sliding window.
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * @brief With indices, applies a 2D max pooling over an input signal composed of several input planes
 * @param[in] ctx Context environment.
 * @param indices It contains the flattened index positions of each maximum value in the max pooling operation. type = [int32, int64].
 * @sa Other parameters refer to diopiMaxPool2d().
 */
DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * @brief compute the backward pass of diopiMaxPool2d().
 * @param[in] grad_output the grad of output. type = [float16, float32].
 * @param[out] grad_input the grad of input. type = [float16, float32].
 * @sa Other parameters refer to diopiMaxPool2d().
 */
DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * @brief Applies a 2D adaptive average pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64]
 * @param output_size an array, the size of the output tensor. type = [int32, int64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief compute the backward pass of diopiAdaptiveAvgPool2d().
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiAdaptiveAvgPool2d().
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input);

/**
 * @brief Applies a 2D adaptive max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float16, float64]
 * @param output_size an array, the size of the output tensor. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief compute the backward pass of diopiAdaptiveMaxPool2d().
 * @param[in] grad_output the grad of output. type = [float32, float16, float64].
 * @param[out] grad_input the grad of input. type = [float32, float16, float64].
 * @sa Other parameters refer to diopiAdaptiveMaxPool2d().
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * @brief Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64].
 * @param p the probability of an element in the input tensor being zeroed out. type = [float32, float64].
 * @param train boolean, whether the module is in training mode. When set to False, the dropout operation will not be performed.
 * @param[out] out the output tensor. type = [float32, float64].
 * @param mask A binary mask tensor of the same shape as the input tensor, where each element's value is either 0 or 1,
 * indicating whether the corresponding neuron at that position is dropped or not. type = [int32].
 */
DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                                    bool train);
/**
 * @brief the in-place version of diopiDropout().
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64].
 * @sa Other parameters refer to diopiDropout().
 */
DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train);

/**
 * @brief Measures the element-wise mean squared error
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64].
 * @param target the target tensor. type = [float32, float64].
 * @param reduction Specifies the reduction to apply to the output.
 * @param[out] out the result tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiReduction_t reduction);
/**
 * @brief Measures the element-wise mean squared error
 * @param[in] input the input tensor. type = [float32, float64].
 * @param grad_output the grad tensor of output. type = [float32, float64].
 * @param target the target tensor. type = [float32, float64].
 * @param reduction Specifies the reduction to apply to the output.
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
                                             diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                     diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha,
                                                     diopiReduction_t reduction);

/**
 * @brief Measures thee Cross Entropy between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param input Input tensor representing the unnormalized scores, often referred to as logits. type = [float32, float64].
 * @param target Target tensor representing the true class index or class probabilities. type = [float32, float64].
 * @param weight  Manual rescaling weight for each class. type = [float32, float64].
 * @param reduction Specifies the reduction to apply to the output.
 * @param ignore_index  Specifies a target value that is to be ignored and does not contribute to the input gradient.
 * Only used when targets are class indices. type = [int64].
 * @param label_smoothing Float value in [0.0, 1.0]. Specifies the amount of smoothing to be applied while computing the loss. type = [float32, float64]
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);
/**
 * @brief compute the backward pass of diopiCrossEntropyLoss().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiCrossEntropyLoss().
 */
DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);

/**
 * @brief Measures thee nll loss between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param input Input tensor, usually representing log probabilities. type = [float32, float64]
 * @param target Target tensor representing class indices, with values in the range of [0, C). type = [int64]
 * @param weight weights manually assigned to each class. type = [float32, float64]
 * @param reduction  Loss reduction mode, which can be none, sum, or mean.
 * @param ignore_index  Specifies a target value to be ignored and does not contribute to the input gradient.
 * This parameter can only be used when the target contains class indices. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index);
/**
 * @brief compute the backward pass of diopiNLLLoss().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiNLLLoss().
 */
DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index);

/**
 * @brief Measures the Binary Cross Entropy between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param pos_weight a weight of positive examples. Must be a vector with length equal to the number of classes. type = [int64].
 * @param reduction Specifies the reduction to apply to the output
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
/**
 * @brief compute the backward pass of diopiBCEWithLogits().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiBCEWithLogits().
 */
DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction);
DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction);

/**
 * \brief Element-wise math functions
 */
DIOPI_API diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiAbs().
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes the absolute value of each element in the input tensor element-wise.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiNeg().
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the negative of the elements of input.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of floor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiSqrt().
 * @param[in] input the input and output tensor and will be stored result tensor, type = [float16, float32]
 */
DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Take the element-wise square root of the input tensor.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32].
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiSin().
 * @param[in] input the input and output tensor and will be stored result tensor,
 * type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Compute the element-wise sine values of the input tensor input.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiAsinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiAsin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiCos().
 * @param[in] input the input and output tensor and will be stored result tensor,
 * type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Compute the element-wise cosine values of the input tensor input.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of tanh.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the hyperbolic tangent of the elements of input.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64].
 * @param[out] out the input tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief Backward pass for tanh.
 * @param[in] grad_output the grad tensor of output.
 * @param output the output tensor. type = [float16, float32, float64].
 * @param[out] grad_input the grad tensor of input. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t output);

/**
 * @brief the in-place version of diopiSigmoid().
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Element-wise applies the sigmoid function to the input tensor input.
 * @param[in] ctx Context environment.
 * @param input the input tensor.type = [float16, float32].
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief compute the backward pass of diopiSigmoid().
 * @param[in] grad_output the grad of output. type = [float16, float32].
 * @param output the output tensor of diopiSigmoid(). type = [float16, float32].
 * @param[out] grad_input the grad of input. type = [float16, float32].
 * @sa Other parameters refer to diopiSigmoid().
 */
DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output);

DIOPI_API diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiExp().
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64]
 */
DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the exponential of the elements of the input tensor input
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64, int16, int32,
 * int64, uint8, int8, bool].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiLog().
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Compute the element-wise natural logarithm of input tensor input.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiLog2().
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Compute the logarithm (base-2) of each element in the input tensor element-wise.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent);

/**
 * @brief Raise each element in the input to the power of the exponent.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @param exponent the value of exponent. type = [int, float].
 * @param[out] out the output tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent);

/**
 * @brief the in-place version of diopiPow().
 * @param[in] input the input tensor andw will be stored result tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @sa Other parameters refer to diopiPow().
 */
DIOPI_API diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent);

/**
 * @brief Raise each element in the input to the power of the corresponding element in exponent.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @param exponent the exponent tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 * @param[out] out the output tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent);

/**
 * @brief the in-place version of diopiPowTensor().
 * @param[in] input the input tensor andw will be stored result tensor. type = [float32, float64, float16].
 * @sa Other parameters refer to diopiPowTensor().
 */
DIOPI_API diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent);

/**
 * @brief This function is used to perform addition operations between tensors.
 * @param[in] ctx Context environment.
 * @param input the first input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool]
 * @param alpha Scaling factor, i.e., the scaling factor of the second tensor.type = [int, float].
 * @param[out] out Output tensor for storing the result of the addition operation. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha);

/**
 * @brief the in-place version of diopiAdd()
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiAdd().
 *
 */
DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha);
/**
 * @brief Add a scalar to a tensor.
 * @param[in] other The scalar value to be added. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @sa Other parameters refer to diopiAdd().
 */
DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha);

/**
 * @brief the in-place version of diopiAddScalar().
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiAddScalar().
 */
DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha);

/**
 * @brief  perform subtraction operations between tensors.
 * @param[in] ctx Context environment.
 * @param input the first input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param alpha Scaling factor, i.e., the scaling factor of the second tensor. type = type = [int, float].
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha);

/**
 * @brief the in-place version of diopiSub().
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiSub().
 */
DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha);

/**
 * @brief sub a scalar to a tensor.
 * @param[in] other The scalar value to be sub. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @sa Other parameters refer to diopiSub().
 */
DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha);

/**
 * @brief the in-place version of diopiSubScalar().
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiSub().
 */
DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha);
/**
 * @brief Multiply tensor input with other (matrix multiplication)
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiMul().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiMul().
 */
DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief Multiply tensor input with other (element-wise multiplication)
 * @param[in] other The scalar value to be added. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @sa Other parameters refer to diopiMul().
 */
DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief the in-place version of diopiMulScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiMul().
 */
DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Divides each element of input tensor by the corresponding element in other tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, dividend. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor, Divisor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param rounding_mode Rounding mode applied to the result, None: no rounding is performed, if both input and other are integer types,
 * the inputs are promoted to the default scalar type; trunc: truncate towards zero; floor: round down towards negative infinity for the result of the division.
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t rounding_mode);

/**
 * @brief the in-place version of diopiDiv().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiDiv().
 */
DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode);

/**
 * @brief Divides each element of input tensor by the scalar element.
 * @param[in] other float scalar, Divisor. type = [int32, int64, float32, float64].
 * @sa Other parameters refer to diopiDiv().
 */
DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      diopiRoundMode_t rounding_mode);

/**
 * @brief the in-place version of diopiDivScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiDivScalar().
 */
DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode);

/**
 * @brief Performs a batch matrix-matrix product of matrices stored in input and mat2.
 * @param[in] ctx Context environment.
 * @param input the first batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param mat2 the second batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

/**
 * @brief Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
 * @param[in] ctx Context environment.
 * @param input the tensor to be added. type = [float16, float32, float64].
 * @param batch1 the first batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param batch2 the second batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param beta the double value beta for multiplier for input.
 * @param alpha the double value alpha for multiplier for batch1@batch2(α).
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                                    diopiConstTensorHandle_t batch2, double beta, double alpha);
                                    /**
 * @brief Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
 * @param[in] ctx Context environment.
 * @param[out] input the input tensor and will be stored result tensor. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiBaddbmm()
 */
DIOPI_API diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2,
                                       double beta, double alpha);

/**
 * @brief Performs the element-wise multiplication.
 * @param[in] ctx Context environment.
 * @param input the input tensor to be added. type = [float16, float32, float64].
 * @param tensor1 the tensor to be multiplied. type = [float16, float32, float64].
 * @param tensor2 the tensor to be multiplied. type = [float16, float32, float64].
 * @param value multiplier tensor1 * tensor2, type = [int, float].
 * @param[out] out the out tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);
/**
 * @brief the in-place version of addcmul.
 * @param[in] ctx Context environment.
 * @param tensor1 the tensor to be multiplied. type = [float16, float32, float64].
 * @param tensor2 the tensor to be multiplied. type = [float16, float32, float64].
 * @param value multiplier for tensor1 * tensor2, type = [int, float].
 * @param[out] input the input tensor to be added and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value);

/**
 * @brief Matrix multiplication. The multiplication rules depend on the dimensions of the input tensors.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64].
 * @param other the second tensor. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Performs the element-wise division.
 * @param[in] ctx Context environment.
 * @param input the input tensor to be added. type = [float16, float32, float64].
 * @param tensor1 the numerator tensor. type = [float16, float32, float64].
 * @param tensor2 the denominator tensor. type = [float16, float32, float64].
 * @param value multiplier for tensor1 / tensor2, type = [int, float].
 * @param[out] out the out tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);
/**
 * @brief the in-place version of addcdiv.
 * @param[in] ctx Context environment.
 * @param tensor1 the numerator tensor. type = [float16, float32, float64].
 * @param tensor2 the denominator tensor. type = [float16, float32, float64].
 * @param value multiplier for tensor1 / tensor2, type = [int, float].
 * @param[out] input the input tensor to be added and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value);

/**
 * @brief Performs matrix multiplication between mat1 and mat2, multiplies the result by scalar value alpha,
 * adds it to input tensor beta x input.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64, float16]].
 * @param mat1 the first martix. type = [float32, float64, float16].
 * @param mat2 the second martix. type = [float32, float64, float16].
 * @param beta scale factor of input. type = [int, float].
 * @param alpha the scaling factor for the multiplication result of the tensors. type = [int, float].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha);

DIOPI_API diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper,
                                     bool checkerror);
DIOPI_API diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t L, bool upper);

DIOPI_API diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b,
                                            diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);
DIOPI_API diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat,
                                                    diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x,
                                                    diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);

/**
 * @brief the in-place version of diopiClampScalar().
 * @param input the input tensor and will be stored result tensor. type = [float32, float64, float16, int16, int32, int64, int8].
 * @sa Other parameters refer to diopiClampScalar()
 */
DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max);

/**
 * @brief the in-place version of diopiClamp().
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64, float16, int16, int32, int64, int8].
 * @sa Other parameters refer to diopiClamp()
 */
DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max);

/**
 * @brief Clamps all elements in input into the range [min, max]
 * @param[in] ctx Context environment.
 * @param input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param min scalar, the lower-bound value. type = [int, float].
 * @param max scalar, the upper-bound value. type = [int, float].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                                        const diopiScalar_t* max);

/**
 * @brief Clamps all elements in input into the range [min, max].
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, int8, uint8]
 * @param min The lower-bound value tensor. type=[float32, float64].
 * @param max The upper-bound value tensor. type=[float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                                  diopiConstTensorHandle_t max);

DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max);
DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max);
DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min);
DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min);
DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min);
DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min);

/**
 * @brief Fills elements of self tensor with value.
 * @param[in] ctx Context environment.
 * @param input the input tensor and output tensor. type = [float32, float64, float16, int16, int32, int64, int8, uint8].
 * @param value the value to fill the tensor with. type = [int, float].
 */
DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value);

/**
 * @brief Computes the element-wise logical AND of the given input tensors.
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tesnor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool]
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiLogicalAnd().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLogicalAnd().
 */
DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes the element-wise logical OR of the given input tensors.
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tesnor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiLogicalOr().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLogicalOr().
 */
DIOPI_API diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

DIOPI_API diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes the bitwise AND of the given input tensors.
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param other the second tesnor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiBitwiseAnd().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseAnd().
 */
DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief Computes the bitwise AND of the given input tensors.
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param other The scalar value to be bitwise and. type = [int, float].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief the in-place version of diopiBitwiseAndScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseAndScalar().
 */
DIOPI_API diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical
 * NOT.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type=[int16, int32, int64, uint8, int8, bool].
 * @param[out] out the result tensor. type=[int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief the in-place version of diopiBitwiseNot.
 * @param[in] ctx Context environment. type=[int16, int32, int64, uint8, int8, bool].
 * @param input the input tensor and will be stored result tensor. type=[int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes equal element-wise comparison with a scalar, ">=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief the in-place version of diopiEqScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiEqScalar().
 */
DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes equal element-wise comparison, "=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor. The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief the in-place version of diopiEq().
 * @param[in] input the input tensor and will be stored result tensor.
 * @sa Other parameters refer to diopiEq().
 */
DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes not equal element-wise comparison with a scalar, "!=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [int, float].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief the in-place version of diopiNeScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiNeScalar().
 */
DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief Computes not equal element-wise comparison, "!=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiNe().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiNe().
 */
DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes greater or equal element-wise comparison with a scalar, ">=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [int, float].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief the in-place version of diopiGeScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiGeScalar().
 */
DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes greater or equal element-wise comparison, ">=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief the in-place version of diopiGe().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiGe().
 */
DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes greater element-wise comparison with a scalar, ">".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [int, float].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief the in-place version of diopiGtScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiGtScalar().
 */
DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes greater element-wise comparison, ">".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief the in-place version of diopiGt().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiGt().
 */
DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes smaller or equal element-wise comparison with a scalar, "<=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief the in-place version of diopiLeScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLeScalar().
 */
DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief Computes smaller or equal element-wise comparison, "<=".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor. The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiLe().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLe().
 */
DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes smaller element-wise comparison with a scalar, "<".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the scalar to be compared. type = [int, float].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true.
 */
DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief the in-place version of diopiLtScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLtScalar().
 */
DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes smaller element-wise comparison, "<".
 * @param[in] ctx Context environment.
 * @param input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param other the second tensor.The dimenson should be same as input tensor.
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief the in-place version of diopiLt().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLt().
 */
DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Returns the mean value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16].
 * @param dim  an array, dimension for reduction. type = [int32, int64].
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim);

/**
 * @brief Returns the sum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16]
 * @param dim an array, dimension for reduction. type = [int32, int64]
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim);

/**
 * @brief Returns the standard derivation of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16].
 * @param dim an array, dimension for reduction. type = [int32, int64].
 * @param unbiased whether to compute the unbiased standard deviation.
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased);

/**
 * @brief Return the minimum value of each row in the input tensor along the given dimension dim.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param dim The dimension along which to reduce. type = [int64]
 * @param[out] min the output tensor, min element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param min_indices the index of the min element. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input,
                                int64_t dim);
/**
 * @brief Returns the minimum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] max the output tensor, min element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input);

/**
 * @brief Return the maximum value of each row in the input tensor along the given dimension dim.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param dim The dimension along which to reduce. type = [int64]
 * @param[out] max the output tensor, max element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param max_indices the index of the max element. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input,
                                int64_t dim);
/**
 * @brief Returns the maximum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[out] max the output tensor, max element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input);

/**
 * @brief Returns True if any element in each row of the tensor in the given dimension dim are True, False otherwise.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type=[bool, float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param dim a int-64 type pointer, the dimension, it can be none.
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [bool, float16, float32, float64, int16,
 * int32, int64, uint8, int8]
 * @param dim a int pointer, the dimension along which the reduction is performed.
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief Applies a softmax function.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64]
 * @param dim The dimension on which to apply the softmax function to the input tensor. type = [int64]
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);
/**
 * @brief compute the backward pass of diopiSoftmax().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param output the output tensor of diopiSoftmax(). type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiNLLLoss().
 */
DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim);

/**
 * @brief Applies a log_softmax function.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64].
 * @param dim the dimension on which to apply the log_softmax function to the input tensor. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);
/**
 * @brief compute the backward pass of diopiLogSoftmax().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param output the output tensor of diopiLogSoftmax(). type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiLogSoftmax().
 */
DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim);

/**
 * @brief Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.
 * @param[in] ctx Context environment.
 * @param[out] out Output tensor.type = [float32, float64]
 * @param input Input tensor.type = [float32, float64]
 * @param indices Array of index tensors.
 * @param nums the int64 value for Number of index tensors.
 */
DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices,
                                  int64_t nums);
/**
 * @brief Performs backward indexing operation on the gradient tensor.
 * @param[in] ctx Context environment.
 * @param[in,out] grad_input Gradient input tensor. type = [float16, float32, float64]
 * @param zeros_like_input Tensor of zeros with the same shape as the input tensor.
 * @param indices Array of index tensors.
 * @param nums the int64 value for Number of index tensors.
 * @param grad Gradient tensor.
 */
DIOPI_API diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                          diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad);
/**
 * @brief Returns a new tensor that indexes the input tensor along dimension dim using the entries in the index tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param dim the dimension along which to index. type = [int64].
 * @param index the index tensor, type = [int32, int64].
 * @param[out] out the output tensor. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                        diopiConstTensorHandle_t index);
/**
 * @brief compute the backward pass of diopiIndexSelect().
 * @param[in] grad_output the grad of output. type = [float32, float64, float16].
 * @param grad the grad tensor of diopiIndexSelect(). type = [float32, float64, float16].
 * @param input_sizes the input tensor sizes of diopiIndexSelect(). type = [int32, int64].
 * @param[out] grad_input the grad of input. type = [float32, float64, float16].
 * @sa Other parameters refer to diopiIndexSelect().
 */
DIOPI_API diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                                diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index);

/**
 * @brief Slices the input tensor along the selected dimension at the given index.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param dim the dimension along which to slice. type = [int64].
 * @param index the index of the slice to return. type = [int64].
 * @param[out] out the output tensor. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index);
/**
 * @brief compute the backward pass of diopiSelect().
 * @param[in] grad_output the grad of output. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param input_sizes the input tensor sizes of diopiSelect(). type = [int32, int16, int64, uint8, int8].
 * @param[out] grad_input the grad of input. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @sa Other parameters refer to diopiSelect().
 */
DIOPI_API diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t index);

/**
 * \brief Embeds the values of the src tensor into input at the given index/dimension. This function returns a tensor with fresh storage; it does not create a
 * view.
 */
DIOPI_API diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src,
                                          int64_t dim, int64_t index);
DIOPI_API diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src,
                                         int64_t dim, int64_t start, int64_t end, int64_t step);
/**
 * \brief Slices the input tensor along the selected dimension at the given index.
 */
DIOPI_API diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start,
                                  int64_t end, int64_t step);
DIOPI_API diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step);

/**
 * \brief Copies elements from source into self tensor at positions where the mask is True.
 */
DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                          diopiConstTensorHandle_t source);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores,
                                double iou_threshold);

/**
 * @brief Returns a tensor containing the indices of all non-zero elements of input.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type=[float32, float16, float64, int16, int32, int64, uint8, int8]
 * @param[out] out the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input);

/**
 * @brief Applies a linear transformation to the incoming data: y=xAT+b.
 * @param[in] ctx Context environment.
 * @param input Input tensor, type = [float16, float32, float64].
 * @param weight weight tensor, type = [float16, float32, float64].
 * @param bias bias tensor, type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                   diopiConstTensorHandle_t bias);
/**
 * @brief compute the backward pass of diopiLinear().
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiLinear().
 */
DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                           diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight);

/**
 * \brief
 */
DIOPI_API diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois,
                                     double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned);

DIOPI_API diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois,
                                             double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels,
                                             int64_t height, int64_t width, int64_t sampling_ratio, bool aligned);

/**
 * @brief Implements stochastic gradient descent optimizer, type=[float32, float16, float64]
 * @param[in] ctx Context environment.
 * @param w the params tensor. type = [float32, float64].
 * @param dw the grad tensor of the params tensor. type = [float32, float64].
 * @param buf the buffer tensor of Momentum. type = [float32, float64].
 * @param lr leaning rate, type = [float32, float64].
 * @param momentum Momentum factor. type = [float32, float64].
 * @param dampening dampening factor. type = [float32, float64].
 * @param weight_decay weight_decay factor. type = [float32, float64].
 * @param nesterov boolean, whether to use Nesterov momentum.
 */
DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                                double dampening, double weight_decay, bool nesterov);

/**
 * @brief Clips gradient norm of an iterable of parameters.
 * @param[in] ctx Context environment.
 * @param grads an iterable of Tensors that will have gradients normalized. type = [float32, float64].
 * @param num_grads the number of grads. type = [int64].
 * @param max_norm max norm of the gradients. type = [float32, float64].
 * @param norm_type type of the used p-norm. Can be ``'inf'`` for infinity norm. type = [float32, float64].
 * @param error_if_nonfinite If True, the operation will return an error if the total norm of the gradients is ``nan`` or ``inf``.
 * @param[out] out total norm of the parameter gradients. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* grads, int64_t num_grads, double max_norm,
                                         double norm_type, bool error_if_nonfinite);

DIOPI_API diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm,
                                             double norm_type);
/**
 * @brief A simple lookup table that looks up embeddings in a fixed dictionary and size.
 * @param[in] ctx Context environment.
 * @param weight the embedding tensor. type = [float32, float64].
 * @param indices the indices tensor. type = [int64].
 * @param padding_idx padding_idx. type = [int64].
 * @param scale_grad_byfreq boolean, whether to scale grad by freq.
 * @param sparse boolean, whether to use sparse update.
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                                      int64_t padding_idx, bool scale_grad_byfreq, bool sparse);
/**
 * @brief compute the backward pass of diopiEmbedding().
 * @param[in] grad the grad of output. type = [float32, float64].
 * @param[out] grad_weight the grad of weight. type = [float32, float64].
 * @sa Other parameters refer to diopiEmbedding().
 */
DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse);

/**
 * @brief Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64, float16, int16, int32,int64, uint8, int8, bool].
 * @param diagonal the diagonal to consider. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32,int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);

/**
 * @brief Concatenates the given sequence of seq tensors in the given dimension.
 * @param[in] ctx Context environment.
 * @param tensors the list of the input tensor list. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 * @param num_inputs the number of input tensor list. type = [int64].
 * @param dim the dimension over which the tensors are concatenated. type = [int64].
 * @param[out] out the output tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim);

/**
 * @brief Splits the tensor into chunks.
 * @param[in] ctx Context environment.
 * @param num_outs the number of output tensor list. type = [int64].
 * @param input the intput tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 * @param splitSizes an array, size of each block or list of sizes for each block. type = [int32, int64].
 * @param dim the dimension along which to split the tensor. type = [int64].
 * @param[out] outs the output tensor list.
 */
DIOPI_API diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input,
                                           const diopiSize_t splitSizes, int64_t dim);

/**
 * @brief Concatenates a sequence of tensors along a new dimension.
 * @param[in] ctx Context environment.
 * @param tensors the list of tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32]
 * @param numTensors the number of tensor list. type = [int64].
 * @param dim  dimension along which to insert. Value must be between 0 and the number of dimensions of the tensor. type = [int64].
 * @param[out] out the output tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim);

/**
 * @brief Sorts the elements of the input tensor along a given dimension in ascending order by value.
 * @param[in] ctx Context environment.
 * @param input the intput tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param dim the dimension to sort along. type = [int64].
 * @param descending boolean, controls the sorting order (ascending or descending).
 * @param stable a boolean pointer, selects a stable sorting algorithm to use,
 * where stable sorting algorithms guarantee that the order of equal elements remains unchanged.
 * @param[out] values the sorted tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param indices the index of corresponding element in the sorted tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                                 bool descending, const bool* stable);

/**
 * @brief Returns the k largest elements of the given input tensor along a given dimension.
 * @param[in] ctx Context environment.
 * @param input the input tesnor.type=[float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param k the k in top-k. type = [int64].
 * @param dim the dimension to sort along. type = [int64].
 * @param largest boolean, whether to return the top k largest elements.
 * @param sorted boolean, whether to return the top k elements in sorted order.
 * @param[out] values the top-k value tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param indices the index of top-k value tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                                 int64_t dim, bool largest, bool sorted);

/**
 * @brief Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1
 * are swapped.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float16, float32, float64, int16,
 * int64, uint8, int8, bool, int32].
 * @param dim0 The first dimension to be transposed. type = [int32, int64].
 * @param dim1 The second dimension to be transposed. type = [int32, int64].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1);

/**
 * @brief Returns a long tensor that has one more dimension with 1 values at the
 *        index of last dimension indicated by the input, and 0 everywhere else.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [int32, int64].
 * @param num_classes The total number of categories. If set to -1, the total number of categories will be inferred as the maximum category value of the input
 * tensor plus one. type = [int64].
 * @param[out] out the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes);

/**
 * @brief Return a tensor of elements selected from either x or y, depending on condition.
 * @param[in] ctx Context environment.
 * @param condition A boolean tensor of the same shape as x and y. For elements/positions where the corresponding value is true,
 * the value from x is returned, otherwise the value from y is returned. type = [uint8, bool].
 * @param input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8, bool]
 * @param other the other tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8, bool]
 * @param[out] out the output tensor. type = [float16, float32, float64, int16,int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t other);

/**
 * @brief Fills elements of self tensor with value where mask is True.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type=[float32, float64, float16].
 * @param mask the boolean mask. type=[bool]
 * @param value the value to fill in with. type=[float32, float64, float16]
 * @param[out] out the result tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value);
/**
 * @brief the in-place version of diopiMaskedFill.
 * @param[in] ctx Context environment.
 * @param input the input tensor, and will be stored result tensor, type=[float32, float64, float16].
 * @param mask the boolean mask. type=[bool].
 * @param value the value to fill in with. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value);
/**
 * @brief Fills elements of self tensor with scalar value where mask is True.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type=[float32, float64, float16].
 * @param mask the boolean mask. type=[bool].
 * @param value the value to fill in with, type = [int, float].
 * @param[out] out the result tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value);
/**
 * @brief the in-place version of diopiMaskedFillScalar.
 * @param[in] ctx Context environment.
 * @param input the input tensor, and will be stored result tensor.
 * @param mask the boolean mask. type=[bool].
 * @param value the value to fill in with, type = [int, float].
 */
DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value);

/**
 * @brief Computes the reciprocal of the elements of input.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float16, float32, float64].
 * @param[out] out the result tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief the in-place version of reciprocal.
 * @param[in] ctx Context environment.
 * @param input the result tensor,  and will be stored result tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Implements AdamW optimizer.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float16, float32, float64].
 * @param grad the grad tensor. type=[float16, float32, float64].
 * @param exp_avg the first momentum is related to the number of iterations, that is, the gradient mean value of the i th iteration. type=[float16, float32,
 * float64].
 * @param exp_avg_sq the second momentum is related to the number of iterations, that is, the mean value of the gradient square of the i iteration.
 * type=[float16, float32, float64].
 * @param max_exp_avg_sq the maximum second momentum. When the parameter 'amsgrad' is true, it will replace the second momentum to participate in the
 * calculation. type=[float16, float32, float64].
 * @param lr learning rate.
 * @param beta1 coefficients used for computing running averages of gradient.
 * @param beta2 square of coefficients.
 * @param eps term added to the denominator to improve numerical stability.
 * @param weight_decay weight decay coefficient.
 * @param step step. type = [int64].
 * @param amsgrad whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_.
 */
DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                  diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int64_t step, bool amsgrad);

/**
 * \brief Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
 */
DIOPI_API diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                            diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups,
                                            diopiSize_t dilation);

/**
 * @brief Extracts sliding local blocks from a batched input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16].
 * @param dim dimension in which unfolding happens. type = [int64].
 * @param size the size of each slice that is unfolded. type = [int64].
 * @param step the step between each slice. type = [int64].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step);
/**
 * @brief Backward pass for diopiUnfold.
 * @param[in] grad_output the grad tensor of output, with the same shape as the forward pass output. type=[float16, float32, float64].
 * @param input_sizes an array, the size of grad_input.
 * @param dim dimension in which unfolding happens. type = [int64].
 * @param size the size of each slice that is unfolded. type = [int64].
 * @param step the step between each slice. type = [int64].
 * @param[out] grad_input the grad tensor of input, with the same shape as the forward pass input. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step);

/**
 * @brief Returns the cumulative sum of elements of input in the dimension dim.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param dim the dimension to do the operation over. type = [int64].
 * @param[out] out the output tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);

/**
 * @brief Computes batched the p-norm distance between each pair of the two collections of row vectors.
 * @param[in] ctx Context environment.
 * @param input1 input tensor of shape B * P * M. type=[float32, float64].
 * @param input2 input tensor of shape B * R * M. type=[float32, float64].
 * @param p double p value for the p-norm distance to calculate between each vector pair.
 * @param compute_mode int64_t* the mode of compute.
 * @param[out] out the output tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p,
                                  const int64_t* compute_mode);
/**
 * @brief Backward pass for cdist.
 * @param[in] grad_output the grad tensor of output, with the same shape as the forward pass output. type=[float32, float64].
 * @param input1 input tensor. type=[float32, float64].
 * @param input2 input tensor. type=[float32, float64].
 * @param p double p value for the p-norm distance to calculate between each vector pair.
 * @param cdist input tensor. type=[float32, float64].
 * @param[out] grad_input the grad tensor of input, with the same shape as the forward pass input. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist);

/**
 * @brief Returns the indices of the maximum values of a tensor across a dimension.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param dim the dimension to do the operation over. type=[int32, int64].
 * @param keepdim whether the output tensor has dim retained or not.
 * @param[out] out the output tensor. type=[int32, int64].
 */
DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim);

/**
 * @brief Implements Adadelta algorithm.
 * @param[in] ctx Context environment.
 * @param[in,out] input The input tensor to be updated. type=[float32, float64, float16]
 * @param[in,out] grad The gradient tensor. type=[float32, float64, float16]
 * @param[in,out] square_avg The moving average of squared gradients. type=[float32, float64, float16]
 * @param[in,out] acc_delta The moving average of squared parameter updates. type=[float32, float64, float16]
 * @param lr the float value for coefficient that scale delta before it is applied to the parameters.
 * @param rho the float value for coefficient used for computing a running average of squared gradients.
 * @param eps the float value for term added to the denominator to improve numerical stability.
 * @param weight_decay the float value for the weight decay (L2 penalty).
 */
DIOPI_API diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                     diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay);

/**
 * @brief Implements Adam optimizer.
 * @param[in] ctx Context environment.
 * @param[in,out] input The input tensor to be updated. type=[float32, float64, float16]
 * @param[in,out] grad The gradient tensor. type=[float32, float64, float16]
 * @param[in,out] exp_avg The exponential moving average of gradients. type=[float32, float64, float16]
 * @param[in,out] exp_avg_sq The exponential moving average of squared gradients. type=[float32, float64, float16]
 * @param[in,out] max_exp_avg_sq The maximum of exponential moving average of squared gradients. type=[float32, float64, float16]
 * @param lr the float value for learning rate.
 * @param beta1 the float value for first coefficients used for computing running averages of gradient and its square.
 * @param beta2 the float value for second coefficients used for computing running averages of gradient and its square.
 * @param eps the float value for term added to the denominator to improve numerical stability
 * @param weight_decay the float value for the weight decay (L2 penalty).
 * @param step the int64 value for the current optimization step.
 * @param amsgrad the bool value for Whether to use the AMSGrad variant of Adam.
 */
DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                 diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                 float weight_decay, int64_t step, bool amsgrad);

/**
 * \brief Implements Rmsprop optimizer.
 * @param[in] ctx Context environment. type = [float16, float32, float64]
 * @param input the input tesor. type = [float16, float32, float64]
 * @param grad the grad tensor. type = [float16, float32, float64]
 * @param square_avg Square average tensor. type = [float16, float32, float64]
 * @param grad_avg Gradient average tensor. type = [float16, float32, float64]
 * @param momentum_buf Momentum buffer tensor. type = [float16, float32, float64]
 * @param lr the float value lr for learning rate.
 * @param alpha the float value alpha for smoothing constant
 * @param eps the float value eps for term added to the denominator to improve numerical stability
 * @param weight_decay the float value weight_decay for weight decay (L2 penalty)
 * @param momentum the float value momentum for momentum factor
 * @param centered the bool value centered if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
 */
DIOPI_API diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                    diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay,
                                    float momentum, bool centered);

/**
 * \brief Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
 */
DIOPI_API diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiReduction_t reduction, double beta);
DIOPI_API diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta);

/**
 * \brief Applies a 3D convolution over an input image composed of several input planes.
 */
DIOPI_API diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups);
DIOPI_API diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups);

/**
 * \brief Applies a 3D max pooling over an input signal composed of several input planes
 */
DIOPI_API diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);
DIOPI_API diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * \brief Applies a 3D adaptive average pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input);

/**
 * \brief Applies a 3D adaptive max pooling over an input signal composed of several input planes.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size);
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * \brief Returns a new 1-D tensor which indexes the input tensor according to the boolean mask.
 */
DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);
DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);

/**
 * \brief Element-wise math functions.
 */
DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

/**
 * @brief Fills the elements of the input tensor with value by selecting the indices in the order given in index.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float16, float32, float64]
 * @param dim int64 value dim for along which to index.
 * @param index indices of self tensor to fill in.type=[int64]
 * @param value Pointer to the scalar value used for filling the elements.type = [int, float].
 * @param[out] out The output tensor. The same shape and type as the input tensor. type=[float16, float32, float64]
 */
DIOPI_API diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                            diopiConstTensorHandle_t index, const diopiScalar_t* value);
/**
 * @brief Fills the elements of the input tensor with value by selecting the indices in the order given in index.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float16, float32, float64]
 * @param dim int64 value dim for along which to index.
 * @param index indices of self tensor to fill in.type=[int64]
 * @param value The value tensor containing the values to fill the elements of the input, same shape and type as the input tensor. type=[float16, float32, float64]
 * @param[out] out The output tensor. The same shape and type as the input tensor. type=[float16, float32, float64]
 */
DIOPI_API diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                      diopiConstTensorHandle_t index, diopiConstTensorHandle_t value);
/**
 * @brief the in-place version of IndexFillScalar().
 * @param[in] ctx Context environment.
 * @sa other parameters refer to IndexFillScalar().
 */
DIOPI_API diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                                               const diopiScalar_t* value);
/**
 * @brief the in-place version of IndexFill().
 * @param[in] ctx Context environment.
 * @sa other parameters refer to IndexFill().
 */
DIOPI_API diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                                         diopiConstTensorHandle_t value);

/**
 * @brief Expand tensor to the same size as out.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 */
DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
 * @param[in] ctx Context environment.
 * @param start the starting value for the set of points. type = [int, float].
 * @param end the ending value for the set of points. type = [int, float].
 * @param steps the number of steps to take from start to end. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64]
 */
DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps);

/**
 * @brief Returns a new tensor with its dimensions permuted.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param dims an array, position order of tensor dimensions during permutation. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * @brief Pads tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float32, float64, float16].
 * @param pad m-elements tuple.
 * @param mode 'constant', 'reflect', 'replicate' or 'circular'.
 * @param value value fill value for 'constant' padding.
 * @param[out] out the output tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode,
                                const double* value);

/**
 * @brief Roll the tensor along the given dimension(s).
 * @param[in] ctx Context environment.
 * @param input the input tensor. type=[float32, float64, float16, bool, int64, int32, int16, int8, uint8, bool].
 * @param shifts The number of places by which the elements of the tensor are shifted.
 * @param dims Axis along which to roll.
 * @param[out] out the output tensor. ype=[float32, float64, float16, bool, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims);

/**
 * \brief Reverse the order of a n-D tensor along given axis in dims.
 */
DIOPI_API diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * @brief Returns the matrix norm or vector norm of a given tensor.
 * @param[in] ctx Context environment.
 * @param input the input tesnor, type=[float32, float64, float16].
 * @param p an array, the order of norm. type = [int, float].
 * @param dim Specifies which dimension or dimensions of input to calculate the norm across.
 * @param[out] out the output tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim);

/**
 * @brief Applies Group Normalization over a mini-batch of inputs.
 * @param[in] ctx Context environment.
 * @param[out] out Output tensor after applying group normalization. type=[float16, float32, float64]
 * @param[out] save_mean Output tensor to store the computed mean for each group. type=[float16, float32, float64]
 * @param[out] save_invstd Output tensor to store the computed inverse standard deviation for each group. type=[float16, float32, float64]
 * @param input Input tensor to be normalized. type=[float16, float32, float64]
 * @param weight Weight tensor for learnable per-channel affine transformation. type=[float16, float32, float64]
 * @param bias Bias tensor for learnable per-channel affine transformation. type=[float16, float32, float64]
 * @param num_groups the int64 value for Number of groups to separate the channels into.
 * @param eps the double value for a value added to the denominator for numerical stability.
 */
DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups,
                                      double eps);
/**
 * @brief Computes the backward pass of Group Normalization.
 * @param[in] ctx Context environment.
 * @param[out] grad_input Gradient of the input tensor. type=[float16, float32, float64]
 * @param[out] grad_weight Gradient of the weight tensor. type=[float16, float32, float64]
 * @param[out] grad_bias Gradient of the bias tensor. type=[float16, float32, float64]
 * @param grad_output Gradient of the output tensor. type=[float16, float32, float64]
 * @param input Input tensor used during the forward pass. type=[float16, float32, float64]
 * @param weight Weight tensor used during the forward pass. type=[float16, float32, float64]
 * @param mean Mean tensor computed during the forward pass. type=[float16, float32, float64]
 * @param rstd Inverse standard deviation tensor computed during the forward pass. type=[float16, float32, float64]
 * @param num_groupsthe int64 value for Number of groups to separate the channels into.
 */
DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd,
                                              int64_t num_groups);

/**
 * @brief Returns the unique elements of the input tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor,type = [int64, float32, float64, float16, int16, int32, uint8, int8, bool]
 * @param dim Specifies the dimension along which the duplicates are removed. It can be None,
 * which means removing duplicates from the entire input tensor.
 * @param sorted boolean, whether to sort the result in ascending order.
 * @param return_counts boolean, whether to return the count tensor
 * @param[out] out the output tensor. type = [int64, float32, float64, float16, int16, int32, uint8, int8, bool].
 * @param indices if none, return new indices of each element in the output tensor. type = [int32, int64].
 * @param counts representing the count of occurrences of each element in the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                                   bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts);

/**
 * \brief Returns the product of all elements in the input tensor.
 */
DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief Computes the Connectionist Temporal Classification loss.
 * @param[in] ctx Context environment.
 * @param log_probs Tensor containing the log probabilities. type=[float32, float64]
 * @param targets Tensor containing the target values.type=[int64]
 * @param input_lengths Tensor containing the lengths of input sequences. type=[int64]
 * @param target_lengths Tensor containing the lengths of target sequences. type=[int64]
 * @param blank Index of the blank label. type=[int64]
 * @param reduction Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the output losses will be divided by the target lengths and then the mean over the batch is taken. Default: 'mean'
 * @param zero_infinity Boolean flag indicating whether to zero out infinite losses. type=[bool]
 * @param[out] neg_log_likelihood Tensor containing the negative log-likelihood values. type=[float32, float64]
 * @param[out] log_alpha Tensor containing the log alpha values. type=[float32, float64]
 * @param[out] Output tensor for storing the computed loss.
 */
DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha,
                                    diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                    diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity);
/**
 * @brief Backward pass for CTCLoss.
 * @param[in] grad_output the grad tensor of output, with the same shape as the forward pass output. type=[float32, float64].
 * @param log_probs Tensor containing the log probabilities. type=[float32, float64]
 * @param targets Tensor containing the target values.type=[int64]
 * @param input_lengths Tensor containing the lengths of input sequences. type=[int64]
 * @param target_lengths Tensor containing the lengths of target sequences. type=[int64]
 * @param blank Index of the blank label. type=[int64]
 * @param reduction Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the output losses will be divided by the target lengths and then the mean over the batch is taken. Default: 'mean'
 * @param zero_infinity Boolean flag indicating whether to zero out infinite losses. type=[bool]
 * @param[out] neg_log_likelihood Tensor containing the negative log-likelihood values. type=[float32, float64]
 * @param[out] log_alpha Tensor containing the log alpha values. type=[float32, float64]
 * @param[out] grad_input the grad tensor of input, with the same shape as the forward pass input. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                            diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood,
                                            diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity);

DIOPI_API diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       diopiConstTensorHandle_t weight);
DIOPI_API diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       const diopiScalar_t* weight);

/**
 * \brief Applies modulus operation.
 */
DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other);

/**
 * @brief Gathers values along an axis specified by dim.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param dim the axis along which to index. type = [int64].
 * @param index the indices of elements to gather. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                   diopiConstTensorHandle_t index);
/**
 * @brief compute the backward pass of diopiGather().
 * @param[in] ctx Context environment.
 * @param grad_output the gradient w.r.t. the output of gather. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] grad_input the gradient w.r.t. the input of gather. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters  refer to diopiGather().
 */
DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index);

/**
 * @brief the in-place version of diopiScatter().
 * @param[in] ctx Context environment.
 * @param input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatter().
 */
DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                                       diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief the in-place version of diopiScatterScalar().
 * @param[in] input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatterScalar().
 */
DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                             diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief Writes all values from the tensor src into input at the indices specified in the index tensor.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param dim the axis along which to index. type = [int64].
 * @param src the source tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param index the indices of elements to scatter. type = [int32, int64].
 * @param reduce the reduce operation. type = [string].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                    diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief Writes all values from the tensor value into input at the indices specified in the index tensor.
 * @param[in] value the value to write into input at the indices. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatter().
 */
DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                          const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce);

/**
 * @brief the in-place version of diopiIndexPut().
 * @param[in] input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiIndexPut().
 */
DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                                        int64_t indices_counts, bool accumulate);
/**
 * @brief Puts values from the tensor values into the tensor input using the indices specified in indices.
 * @param[in] ctx Context environment.
 * @param input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param values the tensor containing the values to copy into input. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param indices the indices into input. type = [int32, int64].
 * @param indices_counts the number of indices. type = [int64].
 * @param accumulate whether to accumulate into input (if true) or perform a copy (if false).
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                                     diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate);

/**
 * @brief Distribution and random numbers.
 * @param[in] ctx Context environment.
 * @param inout the input and output tensor, type = [float32, float64, float16, int64, int32, int16, int8]
 * @param from the lower bound of the random function. type = [int64].
 * @param to a pointer, the upper bound of the random function, it can be none.
 * @param idx idx
 */
DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx);
DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx);

/**
 * @brief Generates a tensor with random binary values drawn from a Bernoulli distribution, element-wise.
 * @param[in] ctx Context environment.
 * @param input The input tensor used for generating random values. type = [float32, float64, float16]
 * @param idx the int64 value for generating random values.
 * @param[out] out The output tensor to store the generated random binary values. type = [float32, float64, float16]
 */
DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx);
/**
 * @brief Generates a tensor with random binary values drawn from a Bernoulli distribution, element-wise, using the given input tensor as an in-place operation.
 * @param[in] ctx Context environment.
 * @param[in,out] inout The input tensor to store the generated random binary values. type = [float32, float64, float16]
 * @param idx the int64 value for generating random values.
 */
DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx);
/**
 * @brief Generates a tensor with random binary values drawn from a Bernoulli distribution, element-wise, using the given scalar probability `p`.
 * @param[in] ctx Context environment.
 * @param p the double value for a pointer, the upper bound of the random function, it can be none.
 * @param idx the int64 value for generating random values.
 * @param out The output tensor to store the generated random binary values. type = [float32, float64, float16]
 */
DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx);

/**
 * @brief Returns a one-dimensional tensor that starts from start, increments by step, and ends at end.
 * @param[in] ctx Context environment.
 * @param start an array, starting value of the resulting tensor. type = [int, float].
 * @param end an array, upper bound of the resulting tensor (exclusive). type = [int, float].
 * @param step an array, difference between adjacent elements of the resulting tensor. type = [int, float].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end,
                                   const diopiScalar_t* step);

/**
 * @brief Randomly generate an integer between 0 and n-1.
 * @param[in] ctx Context environment.
 * @param n the upper bound(excluding), type = [int64].
 * @param idx
 * @param[out] out the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx);

DIOPI_API diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std);
DIOPI_API diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std);
DIOPI_API diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std);
DIOPI_API diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std);
/**
 * @brief Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
 * @param[in] ctx Context environment.
 * @param inout the input and output tensor, type = [float16，float32, float64]
 * @param mean  the double mean value for the mean for all distributions.
 * @param std   the double std value for the std for all distributions.
 */
DIOPI_API diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std);

DIOPI_API diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum);
DIOPI_API diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples,
                                        bool replacement);
/**
 * @brief Applies Layer Normalization over a mini-batch of inputs.
 * type=[float32, float64, float16].
 * @param[in] ctx Context environment.
 * @param save_mean Mean tensor,the mean value for each feature channel of the input tensor. type=[float32, float64, float16].
 * @param save_invstd Backup of inverse standard deviation computed during training. type=[float32, float64, float16].
 * @param input input tensor. type=[float32, float64, float16].
 * @param weight weight tensor. type=[float32, float64, float16].
 * @param bias bias tensor. type=[float32, float64, float16].
 * @param normalized_shape an array, input shape from an expected input of size.
 * @param eps float64 a value added to the denominator for numerical stability.
 * @param[out] out normalized result. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps);
/**
 * @brief Backward pass for diopiLayerNorm. Computes gradients for input, weight, and bias.
 * type=[float32, float64, float16].
 * @param[in] grad_output the grad tensor of output. type=[float32, float64, float16].
 * @param grad_bias the grad of bias. type=[float32, float64, float16].
 * @param grad_weight the grad of weight. type=[float32, float64, float16].
 * @param mean Mean tensor,the mean value for each feature channel of the input tensor. type=[float32, float64, float16].
 * @param rstd Backup of inverse standard deviation computed during training. type=[float32, float64, float16].
 * @param input input tensor. type=[float32, float64, float16].
 * @param weight weight tensor. type=[float32, float64, float16].
 * @param bias bias tensor. type=[float32, float64, float16].
 * @param normalized_shape an array, input shape from an expected input of size.
 * @param[out] grad_input the grad of input. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape);

/**
 * @brief Copies the elements from src into dest tensor.
 * @param[in] ctx Context environment.
 * @param src the source tensor.type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 * @param[out] dest the destination tensor.type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 */
DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest);

/**
 * @brief Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
 * @param[in] ctx Context environment.
 * @param[out] out Output tensor after upsampling. type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 * @param input Input tensor to be upsampled. type= [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 * @param size an array,size of the output tensor.
 */
DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size);
/**
 * @brief Computes the backward pass of nearest neighbor upsampling.
 * @param[in] ctx Context environment.
 * @param[out] grad_input Gradient of the input tensor. type = [float16, float32, float64]
 * @param grad_output Gradient of the output tensor. type = [float16, float32, float64]
 * @param out_size an array,size of the output tensor.
 * @param in_size an array,size of the input tensor.
 * @return Error code indicating the status of the operation.
 */
DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size);
DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char* mode);
DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode);

/**
 * \brief Computes the inverse error function of input tensor.
 */
DIOPI_API diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * \brief Extracts sliding local blocks from a batched input tensor.
 */
DIOPI_API diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                   diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);

/**
 * \brief Combines an array of sliding local blocks into a large containing tensor.
 */
DIOPI_API diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size,
                                   diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);

/**
 * @brief Repeats tensor input along the specified dimensions.
 * @param[in] ctx Context environment.
 * @param input the input tensor, type = [float32, float64].
 * @param repeats_size an integer array containing the number of repetitions needed on each dimension. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size);

DIOPI_API diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiPolar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle);
#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
