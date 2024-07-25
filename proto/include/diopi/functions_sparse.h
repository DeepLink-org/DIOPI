/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_SPARSE_H_
#define _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_SPARSE_H_

#include <diopi/diopirt.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/**
 * @brief           Row Balance Row Major Sequence Reduce SpMM
 * @param[in]       ctx         Context environment.
 * @param[out]      out         Output tensor
 * @param[in]       input       Input tensor
 * @param[in]       mat2        A tensor that stores input matrix data
 */
DIOPI_API diopiError_t diopiSpMM(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_SPARSE_H_
