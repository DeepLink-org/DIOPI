/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief Fused FFN layer.(Act(x * W1) dot (x * W2)) * W3.
 * @param[in] ctx diopi context.
 * @param[out] output : Output tensor.type = [float32]
 * @param[in] input : Input tensor.type = [float32]
 * @param weight1 : Weight1.type = [float32]
 * @param weight2 : Weight2.type = [float32]
 * @param weight3 : Weight3.type = [float32]
 * @param weight3 : Weight3.type = [float32]
 * @param act_type : Type of act which is silu in llama, 0 = silu.type = [int]
 */
DIOPI_API diopiError_t diopiFusedFfn(diopiContextHandle_t ctx, diopiTensorHandle_t output, 
                                     diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight1,
                                     diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3,
                                     const int act_type = 0);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_LMDEPLOY_H_