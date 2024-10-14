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

/**
 * @brief           Fetch-on-demand SpConv
 * @param[in]       ctx             Context environment.
 * @param[in]       in_feat         (N, c) N=# of input points, c = input channels
 * @param[out]      out_feat        (M, o) M=# of output points, o = output channels
 * @param[in]       kernel          (k^3, c, o) for a 3D convolution of length k
 * @param[in]       neighbor_map    (a, 2) the hash table query results from in_coords to out_coords
 */
DIOPI_API diopiError_t diopiSpConv(diopiContextHandle_t ctx, diopiTensorHandle_t out_feat, diopiTensorHandle_t in_feat,
        diopiTensorHandle_t kernel, diopiTensorHandle_t neighbor_map, const int sum_nnz,
        diopiTensorHandle_t neighbor_address, diopiTensorHandle_t q_neighbor_address, const int output_size,
        const int qsum_nnz, const bool transpose, const bool allow_tf32, const bool allow_fp16);
  


// /**
//  * @brief           Fetch-on-demand SpConv
//  * @param[in]       ctx             Context environment.
//  * @param[in]       in_feat         (N, c) N=# of input points, c = input channels
//  * @param[out]      out_feat        (M, o) M=# of output points, o = output channels
//  * @param[in]       kernel          (k^3, c, o) for a 3D convolution of length k
//  * @param[in]       neighbor_map    (a, 2) the hash table query results from in_coords to out_coords
//  */
// DIOPI_API diopiError_t diopiSpConv(diopiContextHandle_t ctx, diopiTensorHandle_t in_feat, diopiTensorHandle_t out_feat, 
//         diopiConstTensorHandle_t kernel, diopiConstTensorHandle_t neighbor_map, const int sum_nnz,
        // diopiTensorHandle_t neighbor_address, diopiTensorHandle_t q_neighbor_address, const int output_size, 
        // const int qsum_nnz, const bool transpose, const bool allow_tf32, const bool allow_fp16);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // _PROJECT_DIOPERATOR_INTERFACE_FUNCTIONS_SPARSE_H_
