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

DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out1, diopiTensorHandle_t out2, diopiConstTensorHandle_t x1,
                                            diopiConstTensorHandle_t x2, diopiConstTensorHandle_t cos, diopiConstTensorHandle_t sin, const bool conj);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_MMCV_H_
