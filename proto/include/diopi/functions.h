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

/**
 * \brief get the vendor's name who implements the functions
 */
DIOPI_RT_API DIOPI_ATTR_WEEK const char* diopiGetVendorName();
DIOPI_RT_API DIOPI_ATTR_WEEK const char* diopiGetImplVersion();
DIOPI_RT_API DIOPI_ATTR_WEEK const char* diopiGetLastErrorString();

/**
 * @brief Applies a 2D convolution over an input image composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float16, float64].
 * @param[in] weight the weight tensor; dimension of kernel_size must match the number of input spatial dimensions.
 * type = [float32, float16, float64].
 * @param[in] bias bias tensor. type = [float32, float16, float64].
 * @param[in] stride an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] padding an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] dilation an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] groups number of groups for grouped convolution. type = [int64].
 * @param[out] out the result tensor. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups);

/**
 * @brief Backward pass for convolution2d. Computes gradients for input, weight, and bias.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output. type = [float32, float16, float64].
 * @param[in] input the input tensor. type = [float32, float16, float64].
 * @param[in] weight the weight tensor; dimension of kernel_size must match the number of input spatial dimensions.
 * @param[in] bias_sizes an array, indicates that a bias was used in the forward pass and contains the shape of the bias. type = [int32, int64].
 * @param[in] stride an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] padding an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] dilation an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] groups number of groups for grouped convolution. type = [int64].
 * @param[out] grad_input the grad of input. type = [float32, float16, float64].
 * @param[out] grad_weight the grad of weight. type = [float32, float16, float64].
 * @param[out] grad_bias the grad of bias. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, int64_t groups);

/**
 * @brief Applies Batch Normalization for each channel across a batch of data.
 * @param[in] ctx Context environment.
 * @param[in] input input tensor. type = [float32, float16, float64].
 * @param[in] weight weight tensor. type = [float32, float16, float64].
 * @param[in] bias bias tensor. type = [float32, float16, float64].
 * @param[in] running_mean weighted average tensor. type = [float32, float16, float64].
 * @param[in] running_var weighted variance tensor. type = [float32, float16, float64].
 * @param[in] training check if in training mode.
 * @param[in] momentum Used to calculate the running mean and variance during runtime. type = [float32, float64]
 * @param[in] eps The value added to the denominator during batch normalization to ensure numerical stability. type = [float32, float64]
 * @param[out] out normalized result. type = [float32, float16, float64].
 * @param[out] save_mean Mean tensor,the mean value for each feature channel of the input tensor. type = [float32, float16, float64].
 * @param[out] save_invstd Backup of inverse standard deviation computed during training. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps);

/**
 * @brief Computes the mean and inverse standard deviation across a batch of data for Synchronized Batch Normalization (SyncBN).
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor. type = [float32, float16, float64].
 * @param[in] eps The value added to the denominator during calculation to ensure numerical stability. type = [float32, float64].
 * @param[out] mean Mean tensor, the computed mean value for each feature channel of the input tensor. type = [float32, float16, float64].
 * @param[out] invstd Inverse standard deviation tensor, the computed inverse standard deviation for each feature channel of the input tensor. type = [float32,
 * float16, float64].
 */
DIOPI_API diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
                                           double eps);

/**
 * @brief Collects and processes statistics across multiple devices for Synchronized Batch Normalization (SyncBN) with consideration of the count of samples in
 * each device.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor. type = [float32, float64].
 * @param[in] mean_all The tensor of aggregated mean values across all devices. type = [float32, float64].
 * @param[in] invstd_all The tensor of aggregated inverse standard deviation values across all devices. type = [float32, float64].
 * @param[in] counts The tensor representing the count of samples in each device. type = [float32, float64].
 * @param[in] momentum Used to calculate the running mean and variance during runtime. type = [float32, float64].
 * @param[in] eps The value added to the denominator during calculation to ensure numerical stability. type = [float32, float64].
 * @param[out] mean Mean tensor, the computed mean value for each feature channel of the input tensor. type = [float32, float64].
 * @param[out] invstd Inverse standard deviation tensor, the computed inverse standard deviation for each feature channel of the input tensor. type = [float32,
 * float16, float64].
 * @param[out] running_mean Updated running mean tensor. type = [float32, float64].
 * @param[out] running_var Updated running variance tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBatchNormGatherStatsWithCounts(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd,
                                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean_all,
                                                           diopiConstTensorHandle_t invstd_all, diopiTensorHandle_t running_mean,
                                                           diopiTensorHandle_t running_var, float momentum, float eps, diopiConstTensorHandle_t counts);

/**
 * @brief Conducts backward pass reduction operations for Synchronized Batch Normalization (SyncBN).
 * @param[in] ctx Context environment.
 * @param[in] grad_out Gradient tensor back-propagated from the downstream layers. type = [float32, float64].
 * @param[in] input Original input tensor used in the forward pass. type = [float32, float64].
 * @param[in] mean Mean tensor computed in the forward pass. type = [float32, float64].
 * @param[in] invstd Inverse standard deviation tensor computed in the forward pass. type = [float32, float64].
 * @param[in] weight Original weight tensor used in the forward pass. type = [float32, float64].
 * @param[in] input_g Flag to indicate whether to compute gradient with respect to input. type = bool.
 * @param[in] weight_g Flag to indicate whether to compute gradient with respect to weight. type = bool.
 * @param[in] bias_g Flag to indicate whether to compute gradient with respect to bias. type = bool.
 * @param[out] sum_dy Tensor for sum of gradients w.r.t output y. type = [float32, float64].
 * @param[out] sum_dy_xmu Tensor for sum of gradients product with (x - mean). type = [float32, float64].
 * @param[out] grad_weight Gradient tensor w.r.t the weight. type = [float32, float64].
 * @param[out] grad_bias Gradient tensor w.r.t the bias. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sum_dy, diopiTensorHandle_t sum_dy_xmu,
                                                    diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_out,
                                                    diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                                    diopiConstTensorHandle_t weight, bool input_g, bool weight_g, bool bias_g);

/**
 * @brief Conducts element-wise operations for the backward pass of Synchronized Batch Normalization (SyncBN).
 * @param[in] ctx Context environment.
 * @param[in] grad_out Gradient tensor back-propagated from the downstream layers. type = [float32, float64].
 * @param[in] input Original input tensor used in the forward pass. type = [float32, float64].
 * @param[in] mean Mean tensor computed in the forward pass. type = [float32, float64].
 * @param[in] invstd Inverse standard deviation tensor computed in the forward pass. type = [float32, float64].
 * @param[in] weight Original weight tensor used in the forward pass. type = [float32, float64].
 * @param[in] sum_dy Tensor for sum of gradients w.r.t output y. type = [float32, float64].
 * @param[in] sum_dy_xmu Tensor for sum of gradients product with (x - mean). type = [float32, float64].
 * @param[in] count The tensor representing the count of samples. type = [int32].
 * @param[out] grad_input Gradient tensor w.r.t the input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBatchNormBackwardElemt(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_out,
                                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t sum_dy, diopiConstTensorHandle_t sum_dy_xmu,
                                                   diopiConstTensorHandle_t count);

/**
 * @brief Conducts element-wise operations for the forward pass of Synchronized Batch Normalization (SyncBN).
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor. type = [float32, float64].
 * @param[in] weight Weight tensor. type = [float32, float64].
 * @param[in] bias Bias tensor. type = [float32, float64].
 * @param[in] mean Mean tensor, the computed mean value for each feature channel of the input tensor. type = [float32, float64].
 * @param[in] invstd Inverse standard deviation tensor, the computed inverse standard deviation for each feature channel of the input tensor. type = [float32,
 * float16, float64].
 * @param[in] eps The value added to the denominator during calculation to ensure numerical stability. type = [float32, float64].
 * @param[out] out Output tensor, the result of batch normalization. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBatchNormElemt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps);

/**
 * @brief compute the backward pass of batch normalization
 * @param[in] ctx Context environment.
 * @param[in] grad_output Gradient of normalized layer output, with the same shape as the forward pass output. type=[float32, float16, float64].
 * @param[in] input input tensor. type = [float32, float16, float64].
 * @param[in] weight weight tensor. type = [float32, float16, float64].
 * @param[in] running_mean weighted average tensor. type = [float32, float16, float64].
 * @param[in] running_var weighted variance tensor. type = [float32, float16, float64].
 * @param[in] save_mean Mean tensor,the mean value for each feature channel of the input tensor. type = [float32, float16, float64].
 * @param[in] save_invstd Backup of inverse standard deviation computed during training. type = [float32, float16, float64].
 * @param[in] training check if in training mode.
 * @param[in] eps The value added to the denominator during batch normalization to ensure numerical stability. type = [float32, float64]
 * @param[out] grad_input Gradient of the input data, with the same shape as the input data. type = [float32, float16, float64].
 * @param[out] grad_weight Gradient of the weight parameter, with the same shape as the weight parameter. type = [float32, float16, float64].
 * @param[out] grad_bias Gradient of the bias parameter, with the same shape as the bias parameter. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var,
                                              diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps);

/**
 * @brief Applies the rectified linear unit function element-wise.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64].
 * @param[out] out the result tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiRelu().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor.type = [float32, float64].
 */
DIOPI_API diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief It clips the tensor values within a range defined by the lower and upper bounds.
 * Any values below the lower bound are set to the lower bound, and any values above the upper bound are set to the upper bound.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor,type = [float32, float64].
 * @param[in] min_val scalar, the lower bound. type = [float32, float64].
 * @param[in] max_val scalar, the upper bound. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val,
                                     const diopiScalar_t* max_val);
/**
 * @brief The in-place version of diopiHardtanh().
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);

/**
 * @brief Compute the backward pass of diopiHardtanhInp().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val);

DIOPI_API diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input);
/**
 * @brief The function thresholds the input tensor by setting elements greater than a given threshold to the threshold value, while leaving elements less than
 * or equal to the threshold unchanged.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64].
 * @param[in] threshold the value to threshold at. type = [float16, float32, float64].
 * @param[in] value the value to replace with. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                                      const diopiScalar_t* value);

/**
 * @brief The in-place version of diopiThreshold().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiThreshold().
 */
DIOPI_API diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value);

/**
 * @brief Compute the backward pass of diopiThreshold().
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @sa Other parameters refer to diopiThreshold().
 */
DIOPI_API diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* threshold);

/**
 * @brief Applies the gaussian error linear unit function element-wise
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64].
 * @param[in] approximate Whether to use an approximate estimation. If it equals to "tanh", it will use an approximate estimation.
 * @param[out] out theout put tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate);
/**
 * @brief Compute the backward pass of diopiGelu().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiHardtanh().
 */
DIOPI_API diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input, const char* approximate);

/**
 * @brief Applies element-wise, LeakyReLU(x) = max(0,x) + negative_slope*min(0,x)
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64].
 * @param[in] negative_slope Controls the angle of the negative slope.
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope);

/**
 * @brief The in-place version of diopiLeakyRelu().
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float32, float64].
 * @sa Other parameters refer to diopiLeakyRelu().
 */
DIOPI_API diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope);

/**
 * @brief Compute the backward pass of diopiLeakyRelu().
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input_is_result boolean. This is a Boolean value indicating whether the input tensor is the result of the Leaky ReLU operation's forward
 * propagation. It is used to optimize memory usage during backpropagation.
 * @param[out] grad_input the grad of input. type = [float32, float64].
 * @sa Other parameters refer to diopiLeakyRelu().
 */
DIOPI_API diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result);

/**
 * @brief Applies 2D average-pooling operation in kH×kW regions by step size sH×sW steps.
 * @param[in] ctx Context environment.
 * @param[in] input input tensor, type = [float32, float64]
 * @param[in] kernel_size an array, the size of the pooling region. type = [int32, int64].
 * @param[in] stride an array, the stride of the pooling operation. type = [int32, int64].
 * @param[in] padding an array. type = [int32, int64].
 * @param[in] ceil_mode boolean, when set to True, uses ceil instead of floor in the formula to compute the output shape.
 * @param[in] count_include_pad boolean, when True, zero-padding will be included in the mean calculation.
 * @param[in] divisor_override If specified, it will be used as the divisor when computing the average pooling,
 *  otherwise the default is to divide by the total number of pooling elements.
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override);

/**
 * @brief Compute the backward pass of diopiAvgPool2d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input input tensor, type = [float32, float64]
 * @param[in] kernel_size an array, the size of the pooling region. type = [int32, int64].
 * @param[in] stride an array, the stride of the pooling operation. type = [int32, int64].
 * @param[in] padding an array. type = [int32, int64].
 * @param[in] ceil_mode boolean, when set to True, uses ceil instead of floor in the formula to compute the output shape.
 * @param[in] count_include_pad boolean, when True, zero-padding will be included in the mean calculation.
 * @param[in] divisor_override If specified, it will be used as the divisor when computing the average pooling,
 *  otherwise the default is to divide by the total number of pooling elements.
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                              bool count_include_pad, const int64_t* divisor_override);

/**
 * @brief Applies a 2D max pooling over an input signal composed of several input planes
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32]
 * @param[in] kernel_size an array, size of the pooling region. type = [int32, int64].
 * @param[in] stride an array, stride of the pooling operation. type = [int32, int64].
 * @param[in] padding  an array, implicit negative infinity padding on both sides of the input tensor, its value should be >= 0 and <= kernel_size / 2. type =
 * [int32, int64].
 * @param[in] dilation an array, spacing between the elements within the sliding window, its value should be greater than 0. type = [int32, int64].
 * @param[in] ceil_mode boolean, if True, use ceil instead of the default floor operation when computing the output shape.
 * This ensures that every element in the input tensor is covered by a sliding window.
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * @brief With indices, applies a 2D max pooling over an input signal composed of several input planes
 * @param[in] ctx Context environment.
 * @param[in] indices It contains the flattened index positions of each maximum value in the max pooling operation. type = [int32, int64].
 * @param[in] input the input tensor. type = [float16, float32]
 * @param[in] kernel_size an array, size of the pooling region. type = [int32, int64].
 * @param[in] stride an array, stride of the pooling operation. type = [int32, int64].
 * @param[in] padding  an array, implicit negative infinity padding on both sides of the input tensor, its value should be >= 0 and <= kernel_size / 2. type =
 * [int32, int64].
 * @param[in] dilation an array, spacing between the elements within the sliding window, its value should be greater than 0. type = [int32, int64].
 * @param[in] ceil_mode boolean, if True, use ceil instead of the default floor operation when computing the output shape.
 * This ensures that every element in the input tensor is covered by a sliding window.
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * @brief Compute the backward pass of diopiMaxPool2d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float16, float32].
 * @param[in] input the input tensor. type = [float16, float32]
 * @param[in] kernel_size an array, size of the pooling region. type = [int32, int64].
 * @param[in] stride an array, stride of the pooling operation. type = [int32, int64].
 * @param[in] padding  an array, implicit negative infinity padding on both sides of the input tensor, its value should be >= 0 and <= kernel_size / 2. type =
 * [int32, int64].
 * @param[in] dilation an array, spacing between the elements within the sliding window, its value should be greater than 0. type = [int32, int64].
 * @param[in] ceil_mode boolean, if True, use ceil instead of the default floor operation when computing the output shape.
 * This ensures that every element in the input tensor is covered by a sliding window.
 * @param[in] indices It contains the flattened index positions of each maximum value in the max pooling operation. type = [int32, int64].
 * @param[out] grad_input the grad of input. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * @brief Applies a 2D adaptive average pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64]
 * @param[in] output_size an array, the size of the output tensor. type = [int32, int64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief Compute the backward pass of diopiAdaptiveAvgPool2d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[in] input the input tensor. type = [float16, float32, float64]
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input);

/**
 * @brief Applies a 2D adaptive max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float16, float64]
 * @param[in] output_size an array, the size of the output tensor. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief With indices, applies a 2D adaptive max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float16, float64]
 * @param[in] output_size an array, the size of the output tensor. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float16, float64].
 * @param[out] indices the max indices along with the outputs.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief Compute the backward pass of diopiAdaptiveMaxPool2d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float16, float64].
 * @param[in] input the input tensor. type = [float32, float16, float64]
 * @param[in] indices the max indices along with the outputs.
 * @param[out] grad_input the grad of input. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * @brief Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64].
 * @param[in] p the probability of an element in the input tensor being zeroed out. type = [float32, float64].
 * @param[in] train boolean, whether the module is in training mode. When set to False, the dropout operation will not be performed.
 * @param[in] generator a pseudorandom number generator for sampling.
 * @param[out] mask A binary mask tensor of the same shape as the input tensor, where each element's value is either 0 or 1,
 * indicating whether the corresponding neuron at that position is dropped or not. type = [int32].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                                    bool train, diopiGeneratorHandle_t generator);
/**
 * @brief The in-place version of diopiDropout().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64].
 * @param[out] mask A binary mask tensor of the same shape as the input tensor, where each element's value is either 0 or 1,
 * indicating whether the corresponding neuron at that position is dropped or not. type = [int32].
 * @param[in] p the probability of an element in the input tensor being zeroed out. type = [float32, float64].
 * @param[in] train boolean, whether the module is in training mode. When set to False, the dropout operation will not be performed.
 * @param[in] generator a pseudorandom number generator for sampling.
 */
DIOPI_API diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                                       diopiGeneratorHandle_t generator);

/**
 * @brief Measures the element-wise mean squared error
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64].
 * @param[in] target the target tensor. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[out] out the result tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiReduction_t reduction);
/**
 * @brief Measures the element-wise mean squared error
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64].
 * @param[in] grad_output the grad tensor of output. type = [float32, float64].
 * @param[in] target the target tensor. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction);

/**
 * \brief Loss used in RetinaNet for dense detection.
 * @param[in] ctx Context environment.
 * @param[in] inputs A float tensor of arbitrary shape. The predictions for each example.
 * @param[in] targets A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class
 and 1 for the positive class).
 * @param[in] alpha Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.
 * @param[in] gamma Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
 * @param[in] reduction ReductionNone: No reduction will be applied to the output;
 ReductionMean: The output will be averaged; ReductionSum: The output will be summed.
 * @param[out] out Loss tensor with the reduction option applied.
 */
DIOPI_API diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
                                             diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction);

/**
 * \brief Compute the backward pass of diopiSigmoidFocalLoss().
 * @param[in] ctx Context environment.
 * @param[in] input A float tensor of arbitrary shape. The predictions for each example.
 * @param[in] target A float tensor with the same shape as inputs. Stores the binary classification label for each element in inputs (0 for the negative class
 and 1 for the positive class).
 * @param[in] grad_output the grad of out.
 * @param[in] alpha Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore.
 * @param[in] gamma Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
 * @param[in] reduction ReductionNone: No reduction will be applied to the output;
 ReductionMean: The output will be averaged; ReductionSum: The output will be summed.
 * @param[out] grad_input the grad of input.
 */
DIOPI_API diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                     diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha,
                                                     diopiReduction_t reduction);

/**
 * @brief Measures thee Cross Entropy between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor representing the unnormalized scores, often referred to as logits. type = [float32, float64].
 * @param[in] target Target tensor representing the true class index or class probabilities. type = [float32, float64].
 * @param[in] weight  Manual rescaling weight for each class. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[in] ignore_index  Specifies a target value that is to be ignored and does not contribute to the input gradient.
 * Only used when targets are class indices. type = [int64].
 * @param[in] label_smoothing Float value in [0.0, 1.0]. Specifies the amount of smoothing to be applied while computing the loss. type = [float32, float64]
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                             diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);
/**
 * @brief Compute the backward pass of diopiCrossEntropyLoss().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input Input tensor representing the unnormalized scores, often referred to as logits. type = [float32, float64].
 * @param[in] target Target tensor representing the true class index or class probabilities. type = [float32, float64].
 * @param[in] weight  Manual rescaling weight for each class. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[in] ignore_index  Specifies a target value that is to be ignored and does not contribute to the input gradient.
 * Only used when targets are class indices. type = [int64].
 * @param[in] label_smoothing Float value in [0.0, 1.0]. Specifies the amount of smoothing to be applied while computing the loss. type = [float32, float64]
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                     diopiReduction_t reduction, int64_t ignore_index, double label_smoothing);

/**
 * @brief Measures thee nll loss between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, usually representing log probabilities. type = [float32, float64]
 * @param[in] target Target tensor representing class indices, with values in the range of [0, C). type = [int64]
 * @param[in] weight weights manually assigned to each class. type = [float32, float64]
 * @param[in] reduction  Loss reduction mode, which can be none, sum, or mean.
 * @param[in] ignore_index  Specifies a target value to be ignored and does not contribute to the input gradient.
 * This parameter can only be used when the target contains class indices. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index);
/**
 * @brief Compute the backward pass of diopiNLLLoss().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input Input tensor, usually representing log probabilities. type = [float32, float64]
 * @param[in] target Target tensor representing class indices, with values in the range of [0, C). type = [int64]
 * @param[in] weight weights manually assigned to each class. type = [float32, float64]
 * @param[in] reduction  Loss reduction mode, which can be none, sum, or mean.
 * @param[in] ignore_index  Specifies a target value to be ignored and does not contribute to the input gradient.
 * This parameter can only be used when the target contains class indices. type = [int64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction, int64_t ignore_index);

/**
 * @brief Measures the Binary Cross Entropy between the target and input probabilities.
 * @param[in] ctx Context environment.
 * @param[in] input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param[in] target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param[in] weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param[in] pos_weight a weight of positive examples. Must be a vector with length equal to the number of classes. type = [int64].
 * @param[in] reduction Specifies the reduction to apply to the output
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);
/**
 * @brief Compute the backward pass of diopiBCEWithLogits().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param[in] target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param[in] weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param[in] pos_weight a weight of positive examples. Must be a vector with length equal to the number of classes. type = [int64].
 * @param[in] reduction Specifies the reduction to apply to the output
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction);

/**
 * @brief Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities.
 * @param[in] ctx Context environment.
 * @param[in] input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param[in] target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param[in] weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                    diopiConstTensorHandle_t weight, diopiReduction_t reduction);

/**
 * @brief Compute the backward pass of diopiBCELoss().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] input Tensor of arbitrary shape as unnormalized scores (often referred to as logits). type = [float32, float64].
 * @param[in] target Tensor of the same shape as input with values between 0 and 1. type = [float32, float64].
 * @param[in] weight a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch. type = [float32, float64].
 * @param[in] reduction Specifies the reduction to apply to the output
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                            diopiReduction_t reduction);

/**
 * @brief Returns a new tensor with the signs of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiAbs().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes the absolute value of each element in the input tensor element-wise.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiNeg().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the negative of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiFloor().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiCeil().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor.
 */
DIOPI_API diopiError_t diopiCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiSqrt().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor, type = [float16, float32]
 */
DIOPI_API diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Take the element-wise square root of the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32].
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiRsqrt().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor, type = [float16, float32]
 */
DIOPI_API diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiSin().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor,
 * type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Compute the element-wise sine values of the input tensor input.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiAsin().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor,
 * type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiAsinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the arcsine of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAsin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiCos().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor and will be stored result tensor,
 * type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Compute the element-wise cosine values of the input tensor input.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiTanh().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the hyperbolic tangent of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64].
 * @param[out] out the input tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief Compute the backward pass for diopiTanh().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output.
 * @param[in] output the output tensor. type = [float16, float32, float64].
 * @param[out] grad_input the grad tensor of input. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t output);

/**
 * @brief Returns a new tensor with the arctangent of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAtan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief The in-place version of diopiAtan().
 * @param[in] ctx Context environment.
 * @param[inout] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAtanInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief The in-place version of diopiSigmoid().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Element-wise applies the sigmoid function to the input tensor input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.type = [float16, float32].
 * @param[out] out the output tensor. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief Compute the backward pass of diopiSigmoid().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float16, float32].
 * @param[in] output the output tensor of diopiSigmoid(). type = [float16, float32].
 * @param[out] grad_input the grad of input. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output);

/**
 * @brief The in-place version of diopiSilu().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 */
DIOPI_API diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known as the swish function.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief Compute the backward pass of diopiSilu().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float16, float32].
 * @param[in] input the input tensor.
 * @param[out] grad_input the grad of input. type = [float16, float32].
 */
DIOPI_API diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiExp().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64]
 */
DIOPI_API diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
/**
 * @brief Returns a new tensor with the exponential of the elements of the input tensor input
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64, int16, int32,
 * int64, uint8, int8, bool].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiLog().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Compute the element-wise natural logarithm of input tensor input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiLog2().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stroed reuslt tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 */
DIOPI_API diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Compute the logarithm (base-2) of each element in the input tensor element-wise.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiLog10.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 */
DIOPI_API diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with the logarithm to the base 10 of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);
DIOPI_API diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief Raise each element in the input to the power of the exponent.
 * @param[in] ctx Context environment.
 * @param[in] input the input value. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @param[in] exponent the value of exponent. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 * @param[out] out the output tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent);

/**
 * @brief Raise each element in the input to the power of the exponent.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @param[in] exponent the value of exponent. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 * @param[out] out the output tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent);

/**
 * @brief The in-place version of diopiPow().
 * @param[in] ctx Context environment.
 * @param[in] exponent the value of exponent. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 * @param[in] input the input tensor andw will be stored result tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent);

/**
 * @brief Raise each element in the input to the power of the corresponding element in exponent.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 * @param[in] exponent the exponent tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 * @param[out] out the output tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent);

/**
 * @brief The in-place version of diopiPowTensor().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor andw will be stored result tensor. type = [float32, float64, float16].
 * @param[in] exponent the exponent tensor. type = [int32, int64, uint8, int8, int16, float32, float64, float16, bool].
 */
DIOPI_API diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent);

/**
 * @brief This function is used to perform addition operations between tensors.
 * \f[out = input + alpha \times other \f]
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool]
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor.type = [float32, float64, int32, int64].
 * @param[out] out Output tensor for storing the result of the addition operation. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha);

/**
 * @brief The in-place version of diopiAdd()
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool]
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor.type = [float32, float64, int32, int64].
 */
DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha);
/**
 * @brief Add a scalar to a tensor.
 * @param[in] other the scalar value to be added. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @sa Other parameters refer to diopiAdd().
 */
DIOPI_API diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha);

/**
 * @brief The in-place version of diopiAddScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other The scalar value to be added. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor.type = [float32, float64, int32, int64].
 */
DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha);

/**
 * @brief  Perform subtraction operations between tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor. type = [float32, float64, int32, int64].
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                const diopiScalar_t* alpha);

/**
 * @brief The in-place version of diopiSub().
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor. type = [float32, float64, int32, int64].
 */
DIOPI_API diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha);

/**
 * @brief Sub a scalar to a tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other The scalar value to be sub. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor. type = [float32, float64, int32, int64].
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      const diopiScalar_t* alpha);

/**
 * @brief The in-place version of diopiSubScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the first input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other The scalar value to be sub. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @param[in] alpha Scaling factor, i.e., the scaling factor of the second tensor. type = [float32, float64, int32, int64].
 */
DIOPI_API diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha);

/**
 * @brief Multiply tensor input with other (matrix multiplication)
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief The in-place version of diopiMul().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Multiply tensor input with other (element-wise multiplication)
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other The scalar value to be multiplied. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiMulScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other The scalar value to be multiplied. type = [float64, float32, float16, int64, int32, int16, int8, uint8].
 */
DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Divides each element of input tensor by the corresponding element in other tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, dividend. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor, Divisor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] rounding_mode Rounding mode applied to the result, None: no rounding is performed, if both input and other are integer types,
 * the inputs are promoted to the default scalar type; trunc: truncate towards zero; floor: round down towards negative infinity for the result of the division.
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t rounding_mode);

/**
 * @brief The in-place version of diopiDiv().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor, Divisor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] rounding_mode Rounding mode applied to the result, None: no rounding is performed, if both input and other are integer types,
 * the inputs are promoted to the default scalar type; trunc: truncate towards zero; floor: round down towards negative infinity for the result of the division.
 */
DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode);

/**
 * @brief Divides each element of input tensor by the scalar element.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other float scalar, Divisor. type = [int32, int64, float32, float64].
 * @param[in] rounding_mode Rounding mode applied to the result, None: no rounding is performed, if both input and other are integer types,
 * the inputs are promoted to the default scalar type; trunc: truncate towards zero; floor: round down towards negative infinity for the result of the division.
 * @param[out] out the output tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      diopiRoundMode_t rounding_mode);

/**
 * @brief The in-place version of diopiDivScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other float scalar, Divisor. type = [int32, int64, float32, float64].
 * @param[in] rounding_mode Rounding mode applied to the result, None: no rounding is performed, if both input and other are integer types,
 * the inputs are promoted to the default scalar type; trunc: truncate towards zero; floor: round down towards negative infinity for the result of the division.
 */
DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode);

/**
 * @brief Broadcast-BLAS functions
 * \f[ out = input @ mat_2 \f]
 * @param[in] ctx Context environment.
 * @param[in] input the first batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param[in] mat2 the second batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

/**
 * @brief Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
 * \f[ out = \beta \times input + \alpha (batch_1 @ batch_2) \f]
 * @param[in] ctx Context environment.
 * @param[in] input the tensor to be added.
 * @param[in] batch1 the first batch of matrices to be multiplied.
 * @param[in] batch2 the second batch of matrices to be multiplied.
 * @param[in] beta multiplier for input.
 * @param[in] alpha multiplier for batch1 and batch2.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                                    diopiConstTensorHandle_t batch2, double beta, double alpha);

/**
 * @brief The in-place version of diopiBaddbmm().
 * @param[in] ctx Context environment.
 * @param[in] input the tensor to be added.
 * @param[in] batch1 the first batch of matrices to be multiplied.
 * @param[in] batch2 the second batch of matrices to be multiplied.
 * @param[in] beta multiplier for input.
 * @param[in] alpha multiplier for batch1 and batch2.
 */
DIOPI_API diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2,
                                       double beta, double alpha);

/**
 * @brief Performs the element-wise multiplication.
 * \f[ out = input + value \times tensor_1 \times tensor_2 \f]
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor to be added. type = [float16, float32, float64].
 * @param[in] tensor1 the tensor to be multiplied. type = [float16, float32, float64].
 * @param[in] tensor2 the tensor to be multiplied. type = [float16, float32, float64].
 * @param[in] value multiplier tensor1 * tensor2, type=[float16, float32, float64].
 * @param[out] out the out tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);
/**
 * @brief The in-place version of diopiAddcmul().
 * @param[in] ctx Context environment.
 * @param[in] tensor1 the tensor to be multiplied. type = [float16, float32, float64].
 * @param[in] tensor2 the tensor to be multiplied. type = [float16, float32, float64].
 * @param[in] value multiplier for tensor1 * tensor2, type=[float16, float32, float64].
 * @param[out] input the input tensor to be added and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value);

/**
 * @brief Matrix multiplication. The multiplication rules depend on the dimensions of the input tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64].
 * @param[in] other the second tensor. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Performs the element-wise division.
 * \f[ out = input + value \times \frac{tensor_1}{tensor_2} \f]
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor to be added. type = [float16, float32, float64].
 * @param[in] tensor1 the numerator tensor. type = [float16, float32, float64].
 * @param[in] tensor2 the denominator tensor. type = [float16, float32, float64].
 * @param[in] value multiplier for tensor1 / tensor2, type=[float16, float32, float64].
 * @param[out] out the out tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value);
/**
 * @brief The in-place version of diopiAddcdiv().
 * @param[in] ctx Context environment.
 * @param[in] tensor1 the numerator tensor. type = [float16, float32, float64].
 * @param[in] tensor2 the denominator tensor. type = [float16, float32, float64].
 * @param[in] value multiplier for tensor1 / tensor2, type=[float16, float32, float64].
 * @param[out] input the input tensor to be added and will be stored result tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value);

/**
 * @brief Performs matrix multiplication between mat1 and mat2, multiplies the result by scalar value alpha,
 * adds it to input tensor beta x input.
 * \f[ out = \beta \times input + \alpha \left ( mat_1 @ mat_2 \right ) \f]
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16]].
 * @param[in] mat1 the first martix. type = [float32, float64, float16].
 * @param[in] mat2 the second martix. type = [float32, float64, float16].
 * @param[in] beta scale factor of input. type = [int32, int64, float32, float64].
 * @param[in] alpha the scaling factor for the multiplication result of the tensors. type = [int32, int64, float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha);

/**
 * @brief Computes the Cholesky decomposition of a symmetric positive-definite matrix A or for batches of symmetric positive-definite matrices.
 * @param[in] ctx Context environment.
 * @param[in] mat the input tensor.
 * @param[in] upper flag that indicates whether to return a upper or lower triangular matrix.
 * @param[in] checkerror controls whether to check the content of infos.
 * @param[out] out the output tensor.
 * @param[out] info stores a positive integer for the corresponding matrix.
 */
DIOPI_API diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper,
                                     bool checkerror);

/**
 * @brief Compute the backward pass of diopiCholesky().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output tensor.
 * @param[in] L the output of diopiCholesky.
 * @param[in] upper flag that indicates whether to return a upper or lower triangular matrix.
 * @param[out] grad_mat the grad of input tensor.
 */
DIOPI_API diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output,
                                             diopiConstTensorHandle_t L, bool upper);

/**
 * @brief Solves a system of equations with a square upper or lower triangular invertible matrix A and multiple right-hand sides bb.
 * @param[in] ctx Context environment.
 * @param[in] b multiple right-hand sides of size (∗,m,k) where ∗ is zero of more batch dimensions.
 * @param[in] mat the input triangular coefficient matrix of size (∗,m,m) where ∗ is zero or more batch dimensions.
 * @param[in] upper whether A is upper or lower triangular.
 * @param[in] transpose solves op(A)X = b where op(A) = A^T if this flag is True, and op(A) = A if it is False.
 * @param[in] unitriangular whether A is unit triangular.
 * @param[out] out the solution X to AX=b.
 * @param[out] cloned_mat a clone of mat.
 */
DIOPI_API diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b,
                                            diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);

/**
 * @brief Compute the backward pass of diopiTriangularSolve().
 * @param[in] ctx Context environment.
 * @param[in] grad_x the grad of X.
 * @param[in] grad_cloned_mat the grad of cloned_mat.
 * @param[in] x the solution X to AX=b.
 * @param[in] b multiple right-hand sides of size (∗,m,k) where ∗ is zero of more batch dimensions.
 * @param[in] mat the input triangular coefficient matrix of size (∗,m,m) where ∗ is zero or more batch dimensions.
 * @param[in] upper whether A is upper or lower triangular.
 * @param[in] transpose solves op(A)X = b where op(A) = A^T if this flag is True, and op(A) = A if it is False.
 * @param[in] unitriangular whether A is unit triangular.
 * @param[out] grad_b the grad of b.
 * @param[out] grad_mat the grad of mat.
 */
DIOPI_API diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat,
                                                    diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x,
                                                    diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular);

/**
 * @brief The in-place version of diopiClampScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min scalar, the lower-bound value. type = [float32, float64].
 * @param[in] max scalar, the upper-bound value. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max);

/**
 * @brief The in-place version of diopiClamp().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min The lower-bound value tensor. type=[float32, float64].
 * @param[in] max The upper-bound value tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max);

/**
 * @brief Clamps all elements in input into the range [min, max]
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min scalar, the lower-bound value. type = [float32, float64].
 * @param[in] max scalar, the upper-bound value. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                                        const diopiScalar_t* max);

/**
 * @brief Clamps all elements in input into the range [min, max].
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, int8, uint8]
 * @param[in] min The lower-bound value tensor. type=[float32, float64].
 * @param[in] max The upper-bound value tensor. type=[float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                                  diopiConstTensorHandle_t max);

/**
 * @brief The in-place version of diopiClampMaxScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] max scalar, the upper-bound value. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max);

/**
 * @brief The in-place version of diopiClampMax().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] max The upper-bound value tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max);

/**
 * @brief The elements in input greater than max will be clamped down to max.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] max scalar, the upper-bound value. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max);

/**
 * @brief The elements in input greater than max will be clamped down to max.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] max The upper-bound value tensor. type=[float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max);

/**
 * @brief The in-place version of diopiClampMinScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min The lower-bound value tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min);

/**
 * @brief The in-place version of diopiClampMin().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min The lower-bound value tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min);

/**
 * @brief The elements in input less than min will be clamped up to min.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min scalar, the lower-bound value. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min);

/**
 * @brief The elements in input less than min will be clamped up to min.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.type = [float32, float64, float16, int16, int32, int64, int8].
 * @param[in] min The lower-bound value tensor. type=[float32, float64].
 * @param[out] out the output tensor. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min);

/**
 * @brief Fills elements of self tensor with value.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and output tensor. type = [float32, float64, float16, int16, int32, int64, int8, uint8].
 * @param[in] value the value to fill the tensor with. type = [float32, float64, float16, int16, int32, int64, int8, uint8].
 */
DIOPI_API diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value);

/**
 * @brief Computes the element-wise logical AND of the given input tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tesnor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool]
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief The in-place version of diopiLogicalAnd().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLogicalAnd().
 */
DIOPI_API diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes the element-wise logical OR of the given input tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tesnor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiLogicalOr().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLogicalOr().
 */
DIOPI_API diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief Computes the element-wise logical NOT of the given input tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief The in-place version of diopiLogicalNot().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiLogicalNot().
 */
DIOPI_API diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes the bitwise AND of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[in] other the second tesnor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiBitwiseAnd().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseAnd().
 */
DIOPI_API diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief Computes the bitwise AND of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[in] other the scalar value to be bitwise and. type = [int16, int32, int64, int8, uint8, bool].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief The in-place version of diopiBitwiseAndScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseAndScalar().
 */
DIOPI_API diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief Computes the bitwise OR of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical OR.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[in] other the second tesnor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiBitwiseOr().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseOr().
 */
DIOPI_API diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief Computes the bitwise OR of input and other. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical AND.
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @param[in] other the scalar value to be bitwise or. type = [int16, int32, int64, int8, uint8, bool].
 * @param[out] out the output tensor. type = [int16, int32, int64, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief The in-place version of diopiBitwiseOrScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [int16, int32, int64, int8, uint8, bool].
 * @sa Other parameters refer to diopiBitwiseOrScalar().
 */
DIOPI_API diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes the bitwise NOT of the given input tensor. The input tensor must be of integral or Boolean types. For bool tensors, it computes the logical
 * NOT.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type=[int16, int32, int64, uint8, int8, bool].
 * @param[out] out the output tensor. type=[int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiBitwiseNot().
 * @param[in] ctx Context environment. type=[int16, int32, int64, uint8, int8, bool].
 * @param[in] input the input tensor and will be stored result tensor. type=[int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Computes equal element-wise comparison with a scalar, "=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor. Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiEqScalar().
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @sa Other parameters refer to diopiEqScalar().
 */
DIOPI_API diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes equal element-wise comparison, "=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor. The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief The in-place version of diopiEq().
 * @param[in] input the input tensor and will be stored result tensor.
 * @sa Other parameters refer to diopiEq().
 */
DIOPI_API diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes not equal element-wise comparison with a scalar, "!=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief The in-place version of diopiNeScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief Computes not equal element-wise comparison, "!=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiNe().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes greater or equal element-wise comparison with a scalar, ">=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiGeScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes greater or equal element-wise comparison, ">=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief The in-place version of diopiGe().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes greater element-wise comparison with a scalar, ">".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiGtScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes greater element-wise comparison, ">".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief The in-place version of diopiGt().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes smaller or equal element-wise comparison with a scalar, "<=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiLeScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */

DIOPI_API diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);
/**
 * @brief Computes smaller or equal element-wise comparison, "<=".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor. The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */

DIOPI_API diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiLe().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor. The dimenson should be same as input tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes smaller element-wise comparison with a scalar, "<".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true.
 */
DIOPI_API diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief The in-place version of diopiLtScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the scalar to be compared. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes smaller element-wise comparison, "<".
 * @param[in] ctx Context environment.
 * @param[in] input the first tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor.
 * @param[out] out the output tensor.Each element has a boolean value, i.e. either false or true. type = [bool].
 */
DIOPI_API diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);
/**
 * @brief The in-place version of diopiLt().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor and will be stored result tensor. type = [float64, float32, float16, int64, int32, int16, int8, uint8, bool].
 * @param[in] other the second tensor.The dimenson should be same as input tensor.
 */
DIOPI_API diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Returns the mean value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16].
 * @param[in] dim  an array, dimension for reduction. type = [int32, int64].
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim);

/**
 * @brief Returns the sum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16]
 * @param[in] dim an array, dimension for reduction. type = [int32, int64]
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim);

/**
 * @brief Returns the standard derivation of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16].
 * @param[in] dim an array, dimension for reduction. type = [int32, int64].
 * @param[in] unbiased whether to compute the unbiased standard deviation.
 * @param[out] out the output tensor depend on dim. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased);

/**
 * @brief Return the minimum value of each row in the input tensor along the given dimension dim.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[in] dim The dimension along which to reduce. type = [int64]
 * @param[out] min the output tensor, min element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] min_indices the index of the min element. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input,
                                int64_t dim);
/**
 * @brief Returns the minimum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] min the output tensor, min element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input);

/**
 * @brief Return the maximum value of each row in the input tensor along the given dimension dim.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[in] dim The dimension along which to reduce. type = [int64]
 * @param[out] max the output tensor, max element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] max_indices the index of the max element. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input,
                                int64_t dim);
/**
 * @brief Returns the maximum value of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[out] max the output tensor, max element. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input);

/**
 * @brief Returns True if any element in each row of the tensor in the given dimension dim are True, False otherwise.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type=[bool, float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param[in] dim a int-64 type pointer, the dimension, it can be none.
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief Returns True if all elements in each row of the tensor in the given dimension dim are True, False otherwise.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [bool, float16, float32, float64, int16,
 * int32, int64, uint8, int8]
 * @param[in] dim a int pointer, the dimension along which the reduction is performed.
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief Applies a softmax function.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64]
 * @param[in] dim The dimension on which to apply the softmax function to the input tensor. type = [int64]
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);
/**
 * @brief Compute the backward pass of diopiSoftmax().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] output the output tensor of diopiSoftmax(). type = [float32, float64].
 * @param[in] dim The dimension on which to apply the softmax function to the input tensor. type = [int64]
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t output, int64_t dim);

/**
 * @brief Applies a log_softmax function.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64].
 * @param[in] dim the dimension on which to apply the log_softmax function to the input tensor. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);

/**
 * @brief Compute the backward pass of diopiLogSoftmax().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64].
 * @param[in] output the output tensor of diopiLogSoftmax(). type = [float32, float64].
 * @param[in] dim the dimension on which to apply the log_softmax function to the input tensor. type = [int64].
 * @param[out] grad_input the grad of input. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                               diopiConstTensorHandle_t output, int64_t dim);

/**
 * \brief Get a output tensor by extracting the corresponding elements from the input tensor according to the given indices along the dimension of the
 * corresponding position.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param[in] indices an array of index tensors. type = [int32, int64, uint8, bool].
 * @param[in] nums the number of index tensors. type = [int64].
 * @param[out] out the output tensor, type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices,
                                  int64_t nums);
/**
 * @brief Compute the backward pass of diopiIndex().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float32, float64, float16].
 * @param[in] indices an array of index tensors. type = [int32, int64, uint8, bool].
 * @param[in] nums the number of index tensors. type = [int64].
 * @param[in] zeros_like_input the zero tensor with the same shape as input. type = [float32, float64, float16].
 * @param[out] grad_input the grad of input. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                          diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad_output);

/**
 * @brief Returns a new tensor that indexes the input tensor along dimension dim using the entries in the index tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param[in] dim the dimension along which to index. type = [int64].
 * @param[in] index the index tensor, type = [int32, int64].
 * @param[out] out the output tensor. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                        diopiConstTensorHandle_t index);
/**
 * @brief Compute the backward pass of diopiIndexSelect().
 * @param[in] ctx Context environment.
 * @param[in] grad the grad tensor of diopiIndexSelect(). type = [float32, float64, float16].
 * @param[in] input_sizes the input tensor sizes of diopiIndexSelect(). type = [int32, int64].
 * @param[in] dim the dimension along which to index. type = [int64].
 * @param[in] index the index tensor, type = [int32, int64].
 * @param[out] grad_input the grad of input. type = [float32, float64, float16].
 */
DIOPI_API diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                                diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index);

/**
 * @brief Slices the input tensor along the selected dimension at the given index.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param[in] dim the dimension along which to slice. type = [int64].
 * @param[in] index the index of the slice to return. type = [int64].
 * @param[out] out the output tensor. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index);
/**
 * @brief Compute the backward pass of diopiSelect().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 * @param[in] input_sizes the input tensor sizes of diopiSelect(). type = [int32, int16, int64, uint8, int8].
 * @param[in] dim the dimension along which to slice. type = [int64].
 * @param[in] index the index of the slice to return. type = [int64].
 * @param[out] grad_input the grad of input. type = [int32, int16, int64, uint8, int8, bool, float32, float64, float16].
 */
DIOPI_API diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t index);

/**
 * \brief Embeds the values of the src tensor into input at the given index. This function returns a tensor with fresh storage; it does not create a view.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] src the tensor to embed into input. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the dimension to insert the slice into. type = [int64].
 * @param[in] index the index to select with. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src,
                                          int64_t dim, int64_t index);
/**
 * \brief Embeds the values of the src tensor into input at the given dimension. This function returns a tensor with fresh storage; it does not create a view.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] src the tensor to embed into input. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the dimension to insert the slice into. type = [int64].
 * @param[in] start  the start index of where to insert the slice. type = [int64].
 * @param[in] end the end index of where to insert the slice. type = [int64].
 * @param[in] step the how many elements to skip in. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src,
                                         int64_t dim, int64_t start, int64_t end, int64_t step);
/**
 * \brief Slices on input tensor input with the begin begin, end end, stride stride, and returns the results in the output tensor output.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim the dimension of input to be sliced.
 * @param[in] start the starting position.
 * @param[in] end the ending position.
 * @param[in] step the step.
 * @param[out] null_out the output tensor.
 */
DIOPI_API diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start,
                                  int64_t end, int64_t step);

/**
 * \brief Compute the backward pass of diopiSlice().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input_sizes an array, the size of input tensor.
 * @param[in] dim the dimension of input to be sliced.
 * @param[in] start the starting position.
 * @param[in] end the ending position.
 * @param[in] step the step.
 * @param[out] grad_input the grad of input tensor.
 */
DIOPI_API diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step);

/**
 * \brief Copies elements from source into self tensor at positions where the mask is True.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] mask the boolean mask. type=[bool].
 * @param[in] source the tensor to copy from. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                          diopiConstTensorHandle_t source);

/**
 * @brief Computes the subset of input tensor boxes based on the confidence and iou threshold.
 * NMS(Non-Maximum Suppression) selects target boxes with high confidence, based on their intersection over union.
 * @param[in] ctx Context environment.
 * @param[in] boxes the boxes tensor. type=[float16, float32, float64].
 * @param[in] confidence the confidence tensor. type=[float16, float32, float64].
 * @param[in] iou_threshold the threshold of iou. type=[double].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t boxes, diopiConstTensorHandle_t confidence,
                                double iou_threshold);

/**
 * @brief Returns a tensor containing the indices of all non-zero elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type=[float32, float16, float64, int16, int32, int64, uint8, int8]
 * @param[out] out the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input);

/**
 * @brief Applies a linear transformation to the incoming data: y=xAT+b.
 * @param[in] ctx Context environment.
 * @param[in] input Input tensor, type = [float16, float32, float64].
 * @param[in] weight weight tensor, type = [float16, float32, float64].
 * @param[in] bias bias tensor, type = [float16, float32, float64].
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                   diopiConstTensorHandle_t bias);
/**
 * @brief Compute the backward pass of diopiLinear().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output. type = [float16, float32, float64].
 * @param[in] input the input tensor. type = [float16, float32, float64].
 * @param[in] weight the weight tensor. type = [float16, float32, float64].
 * @param[out] grad_input the grad of input. type = [float16, float32, float64].
 * @param[out] grad_weight the grad of weight. type = [float16, float32, float64].
 * @param[out] grad_bias the grad of bias. type = [float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                           diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t weight);

/**
 * @brief Compared with Roi Pooling, Roi Align removes the quantization operation and uses bilinear interpolation to obtain the values of pixels with
 * floating-point coordinates, thus transforming the entire feature aggregation process into a continuous operation.
 * @param[in] ctx Context environment.
 * @param[in] input the iput tensor. type=[float16, float32, float64].
 * @param[in] rois the regions of interest (ROIs) tensor. type=[float16, float32, float64].
 * @param[in] spatial_scale a scaling factor that specifies how to map the box coordinates in the origin image to the coordinates in the output. type=[double].
 * @param[in] pooled_height the height of the pooled regions. type=[int64].
 * @param[in] pooled_width the width of the pooled regions. type=[int64].
 * @param[in] sampling_ratio the number of sampling points in the grid used to compute the output. type=[int64].
 * @param[in] aligned a boolean value which determines whether to shift the boxes by 0.5 pixel. type=[bool].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois,
                                     double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned);
/**
 * @brief Compute the backward pass of diopiRoiAlign().
 * @param[in] ctx Context environment.
 * @param[in] grad the grad tensor. type=[float16, float32, float64].
 * @param[in] rois the regions of interest (ROIs) tensor. type=[float16, float32, float64].
 * @param[in] spatial_scale a scaling factor that specifies how to map the box coordinates in the origin image to the coordinates in the output. type=[double].
 * @param[in] pooled_height the height of the pooled regions. type=[int64].
 * @param[in] pooled_width the width of the pooled regions. type=[int64].
 * @param[in] batch_size batch size. type=[int64].
 * @param[in] channels the number of channels. type=[int64].
 * @param[in] height the height of image. type=[int64].
 * @param[in] width the width of image. type=[int64].
 * @param[in] sampling_ratio the number of sampling points in the grid used to compute the output. type=[int64].
 * @param[in] aligned a boolean value which determines whether to shift the boxes by 0.5 pixel. type=[bool].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois,
                                             double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels,
                                             int64_t height, int64_t width, int64_t sampling_ratio, bool aligned);

/**
 * @brief Implements stochastic gradient descent optimizer, type=[float32, float16, float64]
 * @param[in] ctx Context environment.
 * @param[in] w the params tensor. type = [float32, float64].
 * @param[in] dw the grad tensor of the params tensor. type = [float32, float64].
 * @param[in] buf the buffer tensor of Momentum. type = [float32, float64].
 * @param[in] lr leaning rate, type = [float32, float64].
 * @param[in] momentum Momentum factor. type = [float32, float64].
 * @param[in] dampening dampening factor. type = [float32, float64].
 * @param[in] weight_decay weight_decay factor. type = [float32, float64].
 * @param[in] nesterov boolean, whether to use Nesterov momentum.
 */
DIOPI_API diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                                double dampening, double weight_decay, bool nesterov);

/**
 * @brief Clips gradient norm of an iterable of parameters.
 * @param[in] ctx Context environment.
 * @param[in] grads an iterable of Tensors that will have gradients normalized. type = [float32, float64].
 * @param[in] num_grads the number of grads. type = [int64].
 * @param[in] max_norm max norm of the gradients. type = [float32, float64].
 * @param[in] norm_type type of the used p-norm. Can be ``'inf'`` for infinity norm. type = [float32, float64].
 * @param[in] error_if_nonfinite If True, the operation will return an error if the total norm of the gradients is ``nan`` or ``inf``.
 * @param[out] out total norm of the parameter gradients. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* grads, int64_t num_grads, double max_norm,
                                         double norm_type, bool error_if_nonfinite);

/**
 * @brief Perform weight re-normalization in the embedding layer
 * @param[in] ctx Context environment.
 * @param[in] inout the input tensor and will be stored result tensor. type = [float32, float64].
 * @param[in] indices indicate the weights to be updated in the embedding layer. type = [int32, int64].
 * @param[in] max_norm each embedding vector with norm larger than max_norm is renormalized to have norm max_norm. type = [double].
 * @param[in] norm_type the p of the p-norm to compute for the max_norm option. type = [double].
 */
DIOPI_API diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm,
                                             double norm_type);
/**
 * @brief A simple lookup table that looks up embeddings in a fixed dictionary and size. This module is often used to retrieve word embeddings using indices.
 * The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
 * @param[in] ctx Context environment.
 * @param[in] weight the embedding matrix. type = [float32, float64].
 * @param[in] indices the index of weight that corresponds to each row of output. type = [int32, int64].
 * @param[in] padding_idx determine which index of the embedding vector output should be initialized to zero. type = [int64].
 * @param[in] scale_grad_byfreq boolean, whether to scale grad by freq. type = [bool].
 * @param[in] sparse boolean, whether to use sparse update. type = [bool].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                                      int64_t padding_idx, bool scale_grad_byfreq, bool sparse);
/**
 * @brief Compute the backward pass of diopiEmbedding().
 * @param[in] ctx Context environment.
 * @param[in] grad the grad tensor. type = [float32, float64].
 * @param[in] indices the index of each row of output in grad. type = [int32, int64].
 * @param[in] num_weights the number of weight in the embedding matrix. type = [int64].
 * @param[in] padding_idx determine which index of the embedding vector output should be initialized to zero. type = [int64].
 * @param[in] scale_grad_byfreq boolean, whether to scale grad by freq. type = [bool].
 * @param[in] sparse boolean, whether to use sparse update. type = [bool].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                              diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse);

/**
 * @brief Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32,int64, uint8, int8, bool].
 * @param[in] diagonal the diagonal to consider. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32,int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);

/**
 * @brief Concatenates the given sequence of seq tensors in the given dimension.
 * @param[in] ctx Context environment.
 * @param[in] tensors the list of the input tensor list. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 * @param[in] num_inputs the number of input tensor list. type = [int64].
 * @param[in] dim the dimension over which the tensors are concatenated. type = [int64].
 * @param[out] out the output tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim);

/**
 * @brief Splits the tensor into chunks.
 * @param[in] ctx Context environment.
 * @param[in] num_outs the number of output tensor list. type = [int64].
 * @param[in] input the intput tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 * @param[in] splitSizes an array, size of each block or list of sizes for each block. type = [int32, int64].
 * @param[in] dim the dimension along which to split the tensor. type = [int64].
 * @param[out] outs the output tensor list.
 */
DIOPI_API diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input,
                                           const diopiSize_t splitSizes, int64_t dim);

/**
 * @brief Concatenates a sequence of tensors along a new dimension.
 * @param[in] ctx Context environment.
 * @param[in] tensors the list of tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32]
 * @param[in] numTensors the number of tensor list. type = [int64].
 * @param[in] dim  dimension along which to insert. Value must be between 0 and the number of dimensions of the tensor. type = [int64].
 * @param[out] out the output tensor. type = [float32, float16, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim);

/**
 * @brief Sorts the elements of the input tensor along a given dimension in ascending order by value.
 * @param[in] ctx Context environment.
 * @param[in] input the intput tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param[in] dim the dimension to sort along. type = [int64].
 * @param[in] descending boolean, controls the sorting order (ascending or descending).
 * @param[in] stable a boolean pointer, selects a stable sorting algorithm to use,
 * where stable sorting algorithms guarantee that the order of equal elements remains unchanged.
 * @param[out] values the sorted tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] indices the index of corresponding element in the sorted tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                                 bool descending, const bool* stable);

/**
 * @brief Returns the k largest elements of the given input tensor along a given dimension.
 * @param[in] ctx Context environment.
 * @param[in] input the input tesnor.type=[float16, float32, float64, int16, int32, int64, uint8, int8]
 * @param[in] k the k in top-k. type = [int64].
 * @param[in] dim the dimension to sort along. type = [int64].
 * @param[in] largest boolean, whether to return the top k largest elements.
 * @param[in] sorted boolean, whether to return the top k elements in sorted order.
 * @param[out] values the top-k value tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8].
 * @param[out] indices the index of top-k value tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                                 int64_t dim, bool largest, bool sorted);

/**
 * @brief Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1
 * are swapped.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float16, float32, float64, int16,
 * int64, uint8, int8, bool, int32].
 * @param[in] dim0 The first dimension to be transposed. type = [int32, int64].
 * @param[in] dim1 The second dimension to be transposed. type = [int32, int64].
 * @param[out] out the output tensor. type = [float16, float32, float64, int16, int64, uint8, int8, bool, int32].
 */
DIOPI_API diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1);

/**
 * @brief Returns a long tensor that has one more dimension with 1 values at the
 *        index of last dimension indicated by the input, and 0 everywhere else.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [int32, int64].
 * @param[in] num_classes The total number of categories. If set to -1, the total number of categories will be inferred as the maximum category value of the
 * input tensor plus one. type = [int64].
 * @param[out] out the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes);

/**
 * @brief Return a tensor of elements selected from either x or y, depending on condition.
 * @param[in] ctx Context environment.
 * @param[in] condition A boolean tensor of the same shape as x and y. For elements/positions where the corresponding value is true,
 * the value from x is returned, otherwise the value from y is returned. type = [uint8, bool].
 * @param[in] input the input tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8, bool]
 * @param[in] other the other tensor. type = [float16, float32, float64, int16, int32, int64, uint8, int8, bool]
 * @param[out] out the output tensor. type = [float16, float32, float64, int16,int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t other);

/**
 * @brief Fills elements of self tensor with value where mask is True.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type=[float32, float64, float16].
 * @param[in] mask the boolean mask. type=[bool]
 * @param[in] value the value to fill in with. type=[float32, float64, float16]
 * @param[out] out the result tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value);
/**
 * @brief The in-place version of diopiMaskedFill.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, and will be stored result tensor, type=[float32, float64, float16].
 * @param[in] mask the boolean mask. type=[bool].
 * @param[in] value the value to fill in with. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value);
/**
 * @brief Fills elements of self tensor with scalar value where mask is True.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type=[float32, float64, float16].
 * @param[in] mask the boolean mask. type=[bool].
 * @param[in] value the value to fill in with, type=[float32, float64, float16].
 * @param[out] out the result tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value);
/**
 * @brief The in-place version of diopiMaskedFillScalar.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, and will be stored result tensor.
 * @param[in] mask the boolean mask. type=[bool].
 * @param[in] value the value to fill in with, type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value);

/**
 * @brief Computes the reciprocal of the elements of input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float16, float32, float64].
 * @param[out] out the result tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
/**
 * @brief The in-place version of reciprocal.
 * @param[in] ctx Context environment.
 * @param[in] input the result tensor,  and will be stored result tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Implements AdamW optimizer.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float16, float32, float64].
 * @param[in] grad the grad tensor. type=[float16, float32, float64].
 * @param[in] exp_avg the first momentum is related to the number of iterations, that is, the gradient mean value of the i th iteration. type=[float16, float32,
 * float64].
 * @param[in] exp_avg_sq the second momentum is related to the number of iterations, that is, the mean value of the gradient square of the i iteration.
 * type=[float16, float32, float64].
 * @param[in] max_exp_avg_sq the maximum second momentum. When the parameter 'amsgrad' is true, it will replace the second momentum to participate in the
 * calculation. type=[float16, float32, float64].
 * @param[in] lr learning rate.
 * @param[in] beta1 coefficients used for computing running averages of gradient.
 * @param[in] beta2 square of coefficients.
 * @param[in] eps term added to the denominator to improve numerical stability.
 * @param[in] weight_decay weight decay coefficient.
 * @param[in] step step. type = [int64].
 * @param[in] amsgrad whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_.
 */
DIOPI_API diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                  diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                  float weight_decay, int64_t step, bool amsgrad);

/**
 * @brief Applies a 2D transposed convolution operator over an input image composed of several input planes, sometimes also called “deconvolution”.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float16, float64].
 * @param[in] weight the weight tensor; dimension of kernel_size must match the number of input spatial dimensions.
 * type = [float32, float16, float64].
 * @param[in] bias bias tensor. type = [float32, float16, float64].
 * @param[in] stride an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] padding an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] output_padding an array, dimension == number of input spatial dimensions; only supported when transposed is true. type = [int32, int64].
 * @param[in] dilation an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] groups number of groups for grouped convolution. type = [int64].
 * @param[out] out the result tensor. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                            diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups,
                                            diopiSize_t dilation);

/**
 * @brief Backward pass for ConvTranspose2dBackward. Computes gradients for input, weight, and bias.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output. type = [float32, float16, float64].
 * @param[in] bias_sizes an array, indicates that a bias was used in the forward pass and contains the shape of the bias. type = [int32, int64].
 * @param[in] input the input tensor. type = [float32, float16, float64].
 * @param[in] weight the weight tensor; dimension of kernel_size must match the number of input spatial dimensions.
 * @param[in] stride an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] padding an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] output_padding an array, dimension == number of input spatial dimensions; only supported when transposed is true. type = [int32, int64].
 * @param[in] dilation an array with dimension matching the number of input spatial dimensions. type = [int32, int64].
 * @param[in] groups number of groups for grouped convolution. type = [int64].
 * @param[out] grad_input the grad of input. type = [float32, float16, float64].
 * @param[out] grad_weight the grad of weight. type = [float32, float16, float64].
 * @param[out] grad_bias the grad of bias. type = [float32, float16, float64].
 */
DIOPI_API diopiError_t diopiConvTranspose2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                    diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                    diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                    diopiSize_t dilation, diopiSize_t output_padding, int64_t groups);

/**
 * @brief Extracts sliding local blocks from a batched input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16].
 * @param[in] dim dimension in which unfolding happens. type = [int64].
 * @param[in] size the size of each slice that is unfolded. type = [int64].
 * @param[in] step the step between each slice. type = [int64].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step);
/**
 * @brief Compute the backward pass for diopiUnfold().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output, with the same shape as the forward pass output. type=[float16, float32, float64].
 * @param[in] input_sizes an array, the size of grad_input.
 * @param[in] dim dimension in which unfolding happens. type = [int64].
 * @param[in] size the size of each slice that is unfolded. type = [int64].
 * @param[in] step the step between each slice. type = [int64].
 * @param[out] grad_input the grad tensor of input, with the same shape as the forward pass input. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step);

/**
 * @brief Returns the cumulative sum of elements of input in the dimension dim.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the dimension to do the operation over. type = [int64].
 * @param[out] out the output tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim);

/**
 * @brief Computes batched the p-norm distance between each pair of the two collections of row vectors.
 * @param[in] ctx Context environment.
 * @param[in] input1 input tensor of shape B * P * M. type=[float32, float64].
 * @param[in] input2 input tensor of shape B * R * M. type=[float32, float64].
 * @param[in] p p value for the p-norm distance to calculate between each vector pair. type=[double].
 * @param[in] compute_mode the mode of compute.
 * @param[out] out the output tensor. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p,
                                  const int64_t* compute_mode);
/**
 * @brief Compute the backward pass for diopiCdist().
 * @param[in] grad_output the grad tensor of output, with the same shape as the forward pass output. type=[float32, float64].
 * @param[in] input1 input tensor. type=[float32, float64].
 * @param[in] input2 input tensor. type=[float32, float64].
 * @param[in] p p value for the p-norm distance to calculate between each vector pair. type=[double].
 * @param[in] cdist the p-norm distance between input1 and input2. type=[float32, float64].
 * @param[out] grad_input the grad tensor of input, with the same shape as the forward pass input. type=[float32, float64].
 */
DIOPI_API diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                          diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist);

/**
 * @brief Returns the indices of the maximum values of a tensor across a dimension.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the dimension to do the operation over. type=[int32, int64].
 * @param[in] keepdim whether the output tensor has dim retained or not.
 * @param[out] out the output tensor. type=[int32, int64].
 */
DIOPI_API diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim);

/**
 * @brief Implements Adadelta optimizer.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float16, float32, float64].
 * @param[in] grad the grad tensor. type=[float16, float32, float64].
 * @param[in] square_avg the average of squared gradients. type=[float16, float32, float64].
 * @param[in] acc_delta the accumulated delta. type=[float16, float32, float64].
 * @param[in] lr coefficient that scale delta before it is applied to the parameters. type=[float32].
 * @param[in] rho coefficient used for computing a moving average of squared gradients. type=[float32].
 * @param[in] eps term added to the denominator to improve numerical stability. type=[float32].
 * @param[in] weight_decay weight decay coefficient. type=[float32].
 */
DIOPI_API diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                     diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay);

/**
 * @brief Implements Adam optimizer.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float16, float32, float64].
 * @param[in] grad the grad tensor. type=[float16, float32, float64].
 * @param[in] exp_avg the exponentially weighted moving average of gradients. type=[float16, float32,
 * float64].
 * @param[in] exp_avg_sq the exponentially weighted moving average of squared gradients. type=[float16, float32, float64].
 * @param[in] max_exp_avg_sq the maximum values of the exponentially weighted moving average of squared gradients. type=[float16, float32, float64].
 * @param[in] lr learning rate. type=[float32].
 * @param[in] beta1 coefficients used for computing moving averages of gradients. type=[float32].
 * @param[in] beta2 coefficients used for computing moving averages of squared gradients. type=[float32].
 * @param[in] eps term added to the denominator to improve numerical stability. type=[float32].
 * @param[in] weight_decay weight decay coefficient. type=[float32].
 * @param[in] step step. type = [int64].
 * @param[in] amsgrad whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`. type=[bool].
 */
DIOPI_API diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                                 diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps,
                                 float weight_decay, int64_t step, bool amsgrad);

/**
 * @brief Implements Rmsprop optimizer.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float16, float32, float64].
 * @param[in] grad the grad tensor. type=[float16, float32, float64].
 * @param[in] square_avg the average of squared gradients. type=[float16, float32, float64].
 * @param[in] grad_avg the average of gradients. type=[float16, float32, float64].
 * @param[in] momentum_buf the buffer of momentum. type=[float16, float32, float64].
 * @param[in] lr learning rate. type=[float32].
 * @param[in] alpha smoothing constant. type=[float32].
 * @param[in] eps term added to the denominator to improve numerical stability. type=[float32].
 * @param[in] weight_decay weight decay coefficient. type=[float32].
 * @param[in] momentum momentum factor. type = [float32].
 * @param[in] centered if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance. type=[bool].
 */
DIOPI_API diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                                    diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay,
                                    float momentum, bool centered);

/**
 * \brief Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] target the target tensor.
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[in] beta Specifies the threshold at which to change between L1 and L2 loss.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                         diopiReduction_t reduction, double beta);

/**
 * \brief Compute the backward pass of diopiSmoothL1Loss().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[in] target the target tensor.
 * @param[in] reduction Specifies the reduction to apply to the output.
 * @param[in] beta Specifies the threshold at which to change between L1 and L2 loss.
 * @param[out] grad_input the grad of input.
 */
DIOPI_API diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta);

/**
 * \brief Applies a 3D convolution over an input image composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] weight the learnable weights of the module of shape.
 * @param[in] bias the learnable bias of the module of shape (out_channels).
 * @param[in] stride the stride of the window.
 * @param[in] padding Implicit negative infinity padding to be added on all three sides.
 * @param[in] dilation a parameter that controls the stride of elements in the window.
 * @param[in] groups Number of blocked connections from input channels to output channels.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                          diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups);

/**
 * \brief compute the backward pass of diopiConvolution3d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[in] weight the learnable weights of the module of shape.
 * @param[in] bias_sizes the size of bias tensor.
 * @param[in] stride the stride of the window.
 * @param[in] padding Implicit negative infinity padding to be added on all three sides.
 * @param[in] dilation a parameter that controls the stride of elements in the window.
 * @param[in] groups Number of blocked connections from input channels to output channels.
 * @param[out] grad_input the grad of input.
 * @param[out] grad_weight the grad of weight.
 * @param[out] grad_bias the grad of bias.
 */
DIOPI_API diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                                  diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                                  diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                                  diopiSize_t dilation, int64_t groups);

/**
 * \brief Applies a 3D max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] kernel_size the size of the window to take a max over.
 * @param[in] stride the stride of the window.
 * @param[in] padding Implicit negative infinity padding to be added on all three sides.
 * @param[in] dilation a parameter that controls the stride of elements in the window.
 * @param[in] ceil_mode when True, will use ceil instead of floor to compute the output shape.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                      diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * \brief Applies a 3D max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] indices A tensor containing the indices of the maximum elements in the pooling operation.
 * @param[in] input the input tensor.
 * @param[in] kernel_size the size of the window to take a max over.
 * @param[in] stride the stride of the window.
 * @param[in] padding Implicit negative infinity padding to be added on all three sides.
 * @param[in] dilation a parameter that controls the stride of elements in the window.
 * @param[in] ceil_mode when True, will use ceil instead of floor to compute the output shape.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                                 diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode);

/**
 * @brief Compute the backward pass of diopiMaxPool3d
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[in] kernel_size the size of the window to take a max over.
 * @param[in] stride the stride of the window.
 * @param[in] padding Implicit negative infinity padding to be added on all three sides.
 * @param[in] dilation a parameter that controls the stride of elements in the window.
 * @param[in] ceil_mode when True, will use ceil instead of floor to compute the output shape.
 * @param[in] indices A tensor containing the indices of the maximum elements in the pooling operation.
 * @param[out] grad_input the grad of input.
 */
DIOPI_API diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                              diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                              diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices);

/**
 * \brief Applies a 3D adaptive average pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] output_size the target output size.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief Compute the backward pass of diopiAdaptiveAvgPool3d().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[out] grad_input the grad of input.
 */
DIOPI_API diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input);

/**
 * \brief Applies a 3D adaptive max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] output_size the target output size.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * \brief Applies a 3D adaptive max pooling over an input signal composed of several input planes.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] output_size the target output size.
 * @param[out] out the output tensor.
 * @param[out] indices the indices tensor.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices,
                                                         diopiConstTensorHandle_t input, diopiSize_t output_size);

/**
 * @brief Compute the backward pass of diopiAdaptiveMaxPool3d()
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[in] indices the indices tensor.
 * @param[out] grad_input the grad of input.
 */
DIOPI_API diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices);

/**
 * \brief Returns a new 1-D tensor which indexes the input tensor according to the boolean mask.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] mask the tensor containing the binary mask to index with.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);

/**
 * \brief Compute the backward pass of diopiMaskedSelect.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output tensor.
 * @param[in] input the input tensor.
 * @param[in] mask the tensor containing the binary mask to index with.
 * @param[out] grad_input the grad of input tensor.
 */
DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask);

/**
 * \brief Computes the element-wise maximum of input and other.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] other the second input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * \brief Computes the element-wise minimum of input and other.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] other the second input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Performs a matrix multiplication of the matrices input and mat2.
 * If input is a (n×m)(n×m) tensor, mat2 is a (m×p)(m×p) tensor, out will be a (n×p)(n×p) tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the first matrix to be matrix multiplied.
 * @param[in] mat2 the second matrix to be matrix multiplied.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

/**
 * \brief Fills the elements of the input tensor with value by selecting the indices in the order given in index.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim The dimension along which the index is applied.
 * @param[in] index indices of self tensor to fill in.
 * @param[in] value the value to fill with.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                            diopiConstTensorHandle_t index, const diopiScalar_t* value);

/**
 * \brief Fills the elements of the input tensor with value by selecting the indices in the order given in index.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim The dimension along which the index is applied.
 * @param[in] index indices of self tensor to fill in.
 * @param[in] value the value to fill with.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                      diopiConstTensorHandle_t index, diopiConstTensorHandle_t value);

/**
 * \brief The in-place version of diopiIndexFillScalar().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim The dimension along which the index is applied.
 * @param[in] index indices of self tensor to fill in.
 * @param[in] value the value to fill with.
 * @param[out] input the input and output tensor.
 */
DIOPI_API diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                                               const diopiScalar_t* value);

/**
 * \brief The in-place version of diopiIndexFill().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim The dimension along which the index is applied.
 * @param[in] index indices of self tensor to fill in.
 * @param[in] value the value to fill with.
 * @param[out] input the input and output tensor.
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
 * @param[in] start the starting value for the set of points. type = [float32, float64, float16, int16, int32, int64]
 * @param[in] end the ending value for the set of points. type = [float32, float64, float16, int16, int32, int64]
 * @param[in] steps the number of steps to take from start to end. type = [int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64]
 */
DIOPI_API diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps);

/**
 * @brief Returns a new tensor with its dimensions permuted.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool]
 * @param[in] dims an array, position order of tensor dimensions during permutation. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * @brief Pads tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float32, float64, float16].
 * @param[in] pad m-elements tuple.
 * @param[in] mode 'constant', 'reflect', 'replicate' or 'circular'.
 * @param[in] value value fill value for 'constant' padding.
 * @param[out] out the output tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode,
                                const double* value);

/**
 * @brief Roll the tensor along the given dimension(s).
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type=[float32, float64, float16, bool, int64, int32, int16, int8, uint8, bool].
 * @param[in] shifts The number of places by which the elements of the tensor are shifted.
 * @param[in] dims Axis along which to roll.
 * @param[out] out the output tensor. ype=[float32, float64, float16, bool, int64, int32, int16, int8, uint8, bool].
 */
DIOPI_API diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims);

/**
 * \brief Reverse the order of a n-D tensor along given axis in dims.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dims axis to flip on.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims);

/**
 * @brief Returns the matrix norm or vector norm of a given tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tesnor, type=[float32, float64, float16].
 * @param[in] p an array, the order of norm.
 * @param[in] dim Specifies which dimension or dimensions of input to calculate the norm across.
 * @param[out] out the output tensor. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim);

/**
 * \brief Applies Group Normalization over a mini-batch of inputs.
 * @param[in] ctx Context environment.
 * @param[in] save_mean the input tensor mean.
 * @param[in] save_invstd the input tensor rstd.
 * @param[in] input the input tensor.
 * @param[in] weight the weight tensor.
 * @param[in] bias the bias tensor.
 * @param[in] num_groups number of groups to separate the channels into.
 * @param[in] eps a value added to the denominator for numerical stability.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups,
                                      double eps);

/**
 * @brief Compute the backward pass of diopiGroupNorm().
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad of output.
 * @param[in] input the input tensor.
 * @param[in] weight the weight tensor.
 * @param[in] mean the input tensor mean.
 * @param[in] rstd the input tensor rstd.
 * @param[in] num_groups number of groups to separate the channels into.
 * @param[out] grad_input the grad of input.
 * @param[out] grad_weight the grad of weight.
 * @param[out] grad_bias the grad of bias.
 */
DIOPI_API diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd,
                                              int64_t num_groups);

/**
 * @brief Returns the unique elements of the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor,type = [int64, float32, float64, float16, int16, int32, uint8, int8, bool]
 * @param[in] dim Specifies the dimension along which the duplicates are removed. It can be None,
 * which means removing duplicates from the entire input tensor.
 * @param[in] sorted boolean, whether to sort the result in ascending order.
 * @param[in] return_counts boolean, whether to return the count tensor
 * @param[out] out the output tensor. type = [int64, float32, float64, float16, int16, int32, uint8, int8, bool].
 * @param[out] indices if none, return new indices of each element in the output tensor. type = [int32, int64].
 * @param[out] counts representing the count of occurrences of each element in the output tensor. type = [int32, int64].
 */
DIOPI_API diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                                   bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts);

/**
 * \brief Returns the product of all elements in the input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] dim the dimension to reduce.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim);

/**
 * @brief The Connectionist Temporal Classification loss.
 * @param[in] ctx Context environment.
 * @param[in] neg_log_likelihood
 * @param[in] log_alpha
 * @param[in] log_probs The logarithmized probabilities of the outputs.
 * @param[in] targets (N,S) or (sum(target_lengths)). Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
 * @param[in] input_lengths Lengths of the inputs.
 * @param[in] target_lengths Lengths of the targets.
 * @param[in] blank Blank label.
 * @param[in] reduction Specifies the reduction to apply to the output.
 * ReductionNone no reduction will be applied.
 * ReductionMean the output losses will be divided by the target lengths and then the mean over the batch is taken.
 * ReductionSum the output will be summed.
 * @param[in] zero_infinity Whether to zero infinite losses and the associated gradients.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha,
                                    diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                    diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity);

/**
 * @brief compute the backward pass of diopiCTCLoss().
 */
DIOPI_API diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                            diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood,
                                            diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity);
/**
 * @brief Does a linear interpolation of two tensors start (given by input) and end based on a tensor weight and returns the resulting out tensor.
 * \f[
 * out_i​=start_i+weight_i×(end_i−start_i)
 * \f]
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, the tensor with the starting points. type=[float16, float32, float64].
 * @param[in] end the tensor with the ending points. type=[float16, float32, float64].
 * @param[in] weight the tensor weight for the interpolation formula. type=[float16, float32, float64].
 * @param[out] out the output tensor. type=[float16, float32, float64].
 */
DIOPI_API diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       diopiConstTensorHandle_t weight);
/**
 * @brief Does a linear interpolation of two tensors start (given by input) and end based on a scalar weight and returns the resulting out tensor.
 * \f[out_i​=start_i+weight_i×(end_i−start_i)\f]
 * @param[in] weight the scalar weight for the interpolation formula. type=[float16, float32, float64].
 * @sa Other parameters refer to diopiLerpTensor().
 */
DIOPI_API diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       const diopiScalar_t* weight);

/**
 * \brief Applies modulus operation.
 */
/**
 * @brief Computes Python’s modulus operation entrywise. The result has the same sign as the divisor other and its absolute value is less than that of other.
 * @param[in] ctx Context environment.
 * @param[in] input the dividend tensor.
 * @param[in] other the divisor tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other);

/**
 * @brief Computes Python’s modulus operation entrywise. The result has the same sign as the divisor other and its absolute value is less than that of other.
 * @param[in] ctx Context environment.
 * @param[in] input the dividend tensor.
 * @param[in] other the divisor value.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other);

/**
 * @brief Computes Python’s modulus operation entrywise. The result has the same sign as the divisor other and its absolute value is less than that of other.
 * @param[in] ctx Context environment.
 * @param[in] input the dividend value.
 * @param[in] other the divisor tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other);

/**
 * @brief Gathers values along an axis specified by dim.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the axis along which to index. type = [int64].
 * @param[in] index the indices of elements to gather. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                   diopiConstTensorHandle_t index);
/**
 * @brief Compute the backward pass of diopiGather().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] grad_output the gradient w.r.t. the output of gather. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the axis along which to index. type = [int64].
 * @param[in] index the indices of elements to gather. type = [int32, int64].
 * @param[out] grad_input the gradient w.r.t. the input of gather. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index);

/**
 * @brief The in-place version of diopiScatter().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatter().
 */
DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                                       diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief The in-place version of diopiScatterScalar().
 * @param[in] input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatterScalar().
 */
DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                             diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief Writes all values from the tensor src into input at the indices specified in the index tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] dim the axis along which to index. type = [int64].
 * @param[in] src the source tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] index the indices of elements to scatter. type = [int32, int64].
 * @param[in] reduce the reduce operation. type = [string].
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                    diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce);
/**
 * @brief Writes all values from the scalar value into input at the indices specified in the index tensor.
 * @param[in] value the scalar containing values to write into input at the indices. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @sa other parameters refer to diopiScatter().
 */
DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                          const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce);

/**
 * @brief The in-place version of diopiIndexPut().
 * @param[in] ctx Context environment.
 * @param[in] input the input and output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] values the tensor containing the values to copy into input. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] indices the indices into input. type = [int32, int64].
 * @param[in] indices_counts the number of indices. type = [int64].
 * @param[in] accumulate whether to accumulate into input (if true) or perform a copy (if false).
 */
DIOPI_API diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                                        int64_t indices_counts, bool accumulate);
/**
 * @brief Puts values from the tensor values into the tensor input using the indices specified in indices.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] values the tensor containing the values to copy into input. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 * @param[in] indices the indices into input. type = [int32, int64].
 * @param[in] indices_counts the number of indices. type = [int64].
 * @param[in] accumulate whether to accumulate into input (if true) or perform a copy (if false).
 * @param[out] out the output tensor. type = [float32, float64, float16, int16, int32, int64, uint8, int8, bool].
 */
DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                                     diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate);

/**
 * @brief Distribution and random numbers.
 * @param[in] ctx Context environment.
 * @param[in] inout the input and output tensor, type = [float32, float64, float16, int64, int32, int16, int8]
 * @param[in] from the lower bound of the random function. type = [int64].
 * @param[in] to a pointer, the upper bound of the random function, it can be none.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, diopiGeneratorHandle_t generator);

/**
 * @brief Fills self tensor with numbers sampled from the continuous uniform distribution: \f[P(x)= \frac{1}{to-from}\f]
 * @param[in] ctx Context environment.
 * @param[in] inout the input and output tensor, type = [float32, float64, float16, int64, int32, int16, int8]
 * @param[in] from the lower bound of the random function. type = [double].
 * @param[in] to the upper bound of the random function. type = [double].
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator);

/**
 * @brief Draws binary random numbers (0 or 1) from a Bernoulli distribution.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor of probability values for the Bernoulli distribution.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiGeneratorHandle_t generator);

/**
 * @brief The in-place version of diopiBernoulli().
 * @param[in] ctx Context environment.
 * @param[in] inout the input tensor of probability values for the Bernoulli distribution.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiGeneratorHandle_t generator);

/**
 * @brief Draws binary random numbers (0 or 1) from a Bernoulli distribution.
 * @param[in] ctx Context environment.
 * @param[in] p probability values for the Bernoulli distribution.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator);

/**
 * @brief Returns a one-dimensional tensor that starts from start, increments by step, and ends at end.
 * @param[in] ctx Context environment.
 * @param[in] start an array, starting value of the resulting tensor. type = [float32, float64].
 * @param[in] end an array, upper bound of the resulting tensor (exclusive). type = [float32, float64].
 * @param[in] step an array, difference between adjacent elements of the resulting tensor. type = [float32, float64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end,
                                   const diopiScalar_t* step);

/**
 * @brief Randomly generate an integer between 0 and n-1.
 * @param[in] ctx Context environment.
 * @param[in] n the upper bound(excluding), type = [int64].
 * @param[out] out the output tensor. type = [int32, int64].
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator);

/**
 * @brief Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
 * @param[in] ctx Context environment.
 * @param[in] mean the tensor of per-element means.
 * @param[in] std the tensor of per-element standard deviations.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator);

/**
 * @brief Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
 * @param[in] ctx Context environment.
 * @param[in] mean the tensor of per-element means.
 * @param[in] std the tensor of per-element standard deviations.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                               diopiGeneratorHandle_t generator);

/**
 * @brief Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
 * @param[in] ctx Context environment.
 * @param[in] mean the tensor of per-element means.
 * @param[in] std the tensor of per-element standard deviations.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                               diopiGeneratorHandle_t generator);

/**
 * @brief Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.
 * @param[in] ctx Context environment.
 * @param[in] mean the tensor of per-element means.
 * @param[in] std the tensor of per-element standard deviations.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                                         diopiGeneratorHandle_t generator);

/**
 * @brief The in-place version of diopiNormal.
 * @param[in] ctx Context environment.
 * @param[in] mean the tensor of per-element means.
 * @param[in] std the tensor of per-element standard deviations.
 * @param[in] inout the input and output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator);

/**
 * @brief Creates grids of coordinates specified by 1D input tensors.
 * @param[in] ctx Context environment.
 * @param[in] inputs an array of 1D input tensors. type = [float32, float64].
 * @param[in] inputsNum the number of input tensors. type = [int64].
 * @param[out] outs the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum);

/**
 * @brief Returns a tensor where each row contains num_samples indices sampled from the
 * multinomial probability distribution located in the corresponding row of tensor input.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] num_samples number of samples to draw.
 * @param[in] replacement whether to draw with replacement or not.
 * @param[out] out the output tensor.
 * @param[in] generator a pseudorandom number generator for sampling
 */
DIOPI_API diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples,
                                        bool replacement, diopiGeneratorHandle_t generator);
/**
 * @brief Applies Layer Normalization over a mini-batch of inputs.
 * type=[float32, float64, float16].
 * @param[in] ctx Context environment.
 * @param[in] save_mean Mean tensor,the mean value for each feature channel of the input tensor. type=[float32, float64, float16].
 * @param[in] save_invstd Backup of inverse standard deviation computed during training. type=[float32, float64, float16].
 * @param[in] input input tensor. type=[float32, float64, float16].
 * @param[in] weight weight tensor. type=[float32, float64, float16].
 * @param[in] bias bias tensor. type=[float32, float64, float16].
 * @param[in] normalized_shape an array, input shape from an expected input of size.
 * @param[in] eps float64 a value added to the denominator for numerical stability.
 * @param[out] out normalized result. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                                      diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                      diopiSize_t normalized_shape, double eps);
/**
 * @brief Compute the backward pass for diopiLayerNorm(). Computes gradients for input, weight, and bias.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output. type=[float32, float64, float16].
 * @param[in] grad_bias the grad of bias. type=[float32, float64, float16].
 * @param[in] grad_weight the grad of weight. type=[float32, float64, float16].
 * @param[in] mean Mean tensor,the mean value for each feature channel of the input tensor. type=[float32, float64, float16].
 * @param[in] rstd Backup of inverse standard deviation computed during training. type=[float32, float64, float16].
 * @param[in] input input tensor. type=[float32, float64, float16].
 * @param[in] weight weight tensor. type=[float32, float64, float16].
 * @param[in] bias bias tensor. type=[float32, float64, float16].
 * @param[in] normalized_shape an array, input shape from an expected input of size.
 * @param[out] grad_input the grad of input. type=[float32, float64, float16].
 */
DIOPI_API diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                              diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                              diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean,
                                              diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape);

/**
 * @brief Copies the elements from src into dest tensor.
 * @param[in] ctx Context environment.
 * @param[in] src the source tensor.type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 * @param[out] dest the destination tensor.type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 */
DIOPI_API diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest);

/**
 * @brief Performs interpolation operation with nearest methods.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] size output spatial size.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size);

/**
 * @brief Compute the backward pass for diopiUpsampleNearest(). Computes gradients for input.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output.
 * @param[in] out_size output spatial size.
 * @param[in] in_size input spatial size.
 * @param[out] grad_input the grad tensor of input.
 */
DIOPI_API diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                    diopiSize_t out_size, diopiSize_t in_size);

/**
 * @brief Performs interpolation operation with linear methods.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] size output spatial size.
 * @param[in] align_corners Geometrically, we consider the pixels of the input and output as squares rather than points.
 * If set to True, the input and output tensors are aligned by the center points of their corner pixels,
 * preserving the values at the corner pixels.
 * If set to False, the input and output tensors are aligned by the corner points of their corner pixels,
 * and the interpolation uses edge value padding for out-of-boundary values,
 * making this operation independent of input size when scale_factor is kept the same.
 * @param[in] mode The interp mode. type = [char *], "linear", "bilinear", "bicubic", "trilinear".
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
                                           bool align_corners, const char* mode);

/**
 * @brief Compute the backward pass for diopiUpsampleLinear(). Computes gradients for input.
 * @param[in] ctx Context environment.
 * @param[in] grad_output the grad tensor of output.
 * @param[in] out_size output spatial size.
 * @param[in] in_size input spatial size.
 * @param[in] align_corners Geometrically, we consider the pixels of the input and output as squares rather than points.
 * If set to True, the input and output tensors are aligned by the center points of their corner pixels,
 * preserving the values at the corner pixels.
 * If set to False, the input and output tensors are aligned by the corner points of their corner pixels,
 * and the interpolation uses edge value padding for out-of-boundary values,
 * making this operation independent of input size when scale_factor is kept the same.
 * @param[in] mode The interp mode. type = [char *], "linear", "bilinear", "bicubic", "trilinear".
 * @param[out] grad_input the grad tensor of input.
 */
DIOPI_API diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                   diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode);

/**
 * \brief Computes the inverse error function of input tensor.
 */
DIOPI_API diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);
DIOPI_API diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * \brief Extracts sliding local blocks from a batched input tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] kernel_size the size of the sliding blocks.
 * @param[in] dilation  a parameter that controls the stride of elements within the neighborhood.
 * @param[in] padding implicit zero padding to be added on both sides of input.
 * @param[in] stride the stride of the sliding blocks in the input spatial dimensions.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
                                   diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);

/**
 * \brief Combines an array of sliding local blocks into a large containing tensor.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] output_size the shape of the spatial dimensions of the output.
 * @param[in] kernel_size the size of the sliding blocks.
 * @param[in] dilation  a parameter that controls the stride of elements within the neighborhood.
 * @param[in] padding implicit zero padding to be added on both sides of input.
 * @param[in] stride the stride of the sliding blocks in the input spatial dimensions.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size,
                                   diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride);

/**
 * @brief Repeats tensor input along the specified dimensions.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64].
 * @param[in] repeats_size an integer array containing the number of repetitions needed on each dimension. type = [int32, int64].
 * @param[out] out the output tensor. type = [float32, float64].
 */
DIOPI_API diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size);

/**
 * @brief Returns a Tensor with same dtype as the Tensor out.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

DIOPI_API diopiError_t diopiPolar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle);

/**
 * @brief Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input,
 * the other elements of the result tensor out are set to 0.
 *
 * The upper triangular part of the matrix is defined as the elements on and above the diagonal.
 *
 * The argument diagonal controls which diagonal to consider.
 * If diagonal = 0, all elements on and above the main diagonal are retained.
 * A positive value excludes just as many diagonals above the main diagonal,
 * and similarly a negative value includes just as many diagonals below the main diagonal.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] diagonal the diagonal to consider.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal);

/**
 * @brief The in-place version of diopiTriu().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[in] diagonal the diagonal to consider.
 */
DIOPI_API diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal);

/**
 * @brief This function is an extension of torch.sign() to complex tensors.
 * It computes a new tensor whose elements have the same angles as the corresponding elements
 * of input and absolute values (i.e. magnitudes) of one for complex tensors
 * and is equivalent to torch.sign() for non-complex tensors.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] out the output tensor.
 */
DIOPI_API diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief The in-place version of diopiSgn().
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 */
DIOPI_API diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input);

/**
 * @brief Returns a new tensor with boolean elements representing if each element of input is NaN or not. Complex values are considered NaN when either their
 * real and/or imaginary part is NaN.
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor, type = [float32, float64, float16, bool, int64, int32, int16, int8, uint8].
 * @param[out] out the output tensor. type = [bool].
 */
DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

/**
 * @brief Computes the QR decomposition of a matrix.
 * @param[in] ctx Context environment.
 * @param[in] A  tensor of shape (*, m, n) where * is zero or more batch dimensions., type = [float64, float32].
 * @param[in] mode  one of ‘reduced’, ‘complete’, ‘r’. Controls the shape of the returned tensors. Default: ‘reduced’.
 * @param[out] Q the output tensor. type = [float64, float32].
 * @param[out] R the output tensor. type = [float64, float32].
 */
DIOPI_API diopiError_t diopiLinalgQR(diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char* mode, diopiTensorHandle_t Q, diopiTensorHandle_t R);

/**
 * @brief Returns the maximum value of each slice of the input tensor in the given dimension(s) dim.
 * @param[in] ctx Context environment.
 * @param[in] self the input tensor. type = [float64, float32, float16, int16, int32, int64, int8, uint8]
 * @param[in] dim (int or tuple of ints) – the dimension or dimensions to reduce.
 * @param[in] keepdim whether the output tensor has dim retained or not.type = [bool].
 * @param[out] out the output tensor. type = [float64, float32, float16, int16, int32, int64, int8, uint8]
 */
DIOPI_API diopiError_t diopiAmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t self, diopiSize_t dim, bool keepdim);

// this contiguous func is temporary, please do not use.
DIOPI_API diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat);

/**
 * @brief          Check inf/NaN and unscale gradients for AMP GradScaler.
 * @details        Multiplies each tensor in scaled_grads by inv_scale in-place.
 *                 If any element of any tensor in scaled_grads is inf or NaN,
 *                 sets found_inf to 1.0.
 * @param[in,out]  scaled_grads      Array of scaled gradient tensors. May
 *                                   contain infs or NaNs.
 * @param[in]      num_scaled_grads  Size of the tensor array @p scaled_grads.
 * @param[out]     found_inf         A single-element float32 tensor to which
 *                                   1.0 will be written if any gradient contain
 *                                   infs/nans. Pre-zeroing found_inf, if
 *                                   appropriate, is the responsibility of the
 *                                   caller.
 * @param[in]      inv_scale         A single-element float32 tensor, storing
 *                                   the inverse of the scale factor by which @p
 *                                   scaled_grads are currently multiplied.
 */
DIOPI_API diopiError_t diopiAmpForeachNonFiniteCheckAndUnscaleInp(diopiContextHandle_t ctx, diopiTensorHandle_t* scaled_grads, int64_t num_scaled_grads,
                                                                  diopiTensorHandle_t found_inf, diopiConstTensorHandle_t inv_scale);

/**
 * @brief          Updates the scale tensor in place for AMP GradScaler.
 * @param[in,out]  current_scale    A one-element float32 tensor containing the
 *                                  scale value.
 * @param[in,out]  growth_tracker   A one-element int32 tensor containing the
 *                                  number of recent consecutive unskipped
 *                                  steps.
 * @param[in]      found_inf        A one-element float32 tensor. If > 0,
 *                                  indicates that infs/NaNs were found by the
 *                                  relevant prior
 *                                  #diopiAmpForeachNonFiniteCheckAndUnscaleInp
 *                                  call, and 0 if no infs/NaNs were found.
 * @param[in]      growth_factor    Multiplier if no infs/NaNs were found
 *                                  (typically slightly > 1).
 * @param[in]      backoff_factor   Multiplier if infs/NaNs were found
 *                                  (typically 0.5).
 * @param[in]      growth_interval  Number of consecutive unskipped steps that
 *                                  must occur for current_scale to be
 *                                  multiplied by growth_factor.
 * @see https://github.com/DeepLink-org/pytorch/blob/main/aten/src/ATen/native/cuda/AmpKernels.cu#L181
 */
DIOPI_API diopiError_t diopiAmpUpdateScaleInp(diopiContextHandle_t ctx, diopiTensorHandle_t current_scale, diopiTensorHandle_t growth_tracker,
                                              diopiConstTensorHandle_t found_inf, double scale_growth_factor, double scale_backoff_factor,
                                              int32_t growth_interval);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_H_
