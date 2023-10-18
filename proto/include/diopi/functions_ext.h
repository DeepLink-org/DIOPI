/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_EXT_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief Apply rotary embedding operation to an input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out The output tensor containing the rotary embeddings. type = [float32, float16, float64].
 * @param[in] x The input tensor which rotary embedding will be applied. type = [float32, float16, float64].
 * @param[in] cos The cosine values. type = [float32, float16, float64].
 * @param[in] sin The sine values. type = [float32, float16, float64].
 * @param[in] conj bool: If `false`, computes regular rotary embeddings. If `true`, computes the complex conjugate of the rotary embeddings.
 */
DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj);

/**
 * @brief Apply Root Mean Square (RMS) Normalization to the input tensor.
 * @param[in] ctx Context environment.
 * @param[out] out the output tensor containing the normalized values. type = [float32, float16, float64].
 * @param[in] invRMS The tensor containing the inverse of root mean square. type = [float32, float16, float64].
 * @param[in] input The input tensor to be normalized. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization.
 * @param[in] weight The gain parameter used to re-scale the standardized summed inputs type = [float32, float16, float64].
 * @param[in] bias The bias tensor for the normalization. type = [float32, float16, float64].
 * @param[in] eps A small value to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRMS, diopiConstTensorHandle_t input,
                                    diopiSize_t normalized_shape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps);

/**
 * @brief Compute the backward pass for Root Mean Square (RMS) Normalization.
 * @param[in] ctx Context environment.
 * @param[out] gradInput The gradient of the input tensor. type = [float32, float16, float64].
 * @param[out] gradWeight The gradient of the weight parameter. type = [float32, float16, float64].
 * @param[out] gradBias The gradient of the bias parameter. type = [float32, float16, float64].
 * @param[in] gradOutput The gradient of the output from the forward pass. type = [float32, float16, float64].
 * @param[in] input The input tensor used in the forward pass. type = [float32, float16, float64].
 * @param[in] weight The weight parameter used in the forward pass. type = [float32, float16, float64].
 * @param[in] bias The bias used in the forward pass. type = [float32, float16, float64].
 * @param[in] invRMS The inverse of the root mean square values computed in the forward pass. type = [float32, float16, float64].
 * @param[in] normalized_shape The shape of the normalization.
 * @param[in] eps A small value used in the computation to avoid division by zero. type = [float64].
 */
DIOPI_API diopiError_t diopiRMSNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t invRMS,
                                            diopiSize_t normalized_shape, double eps);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_
