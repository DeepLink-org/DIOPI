/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_sparse.h>

#include "../helper.hpp"
#include "../sparse_kernel.h"

namespace impl {
namespace cuda {

diopiError_t diopiSpMM(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    impl::aten::setCurStream(ctx);
    auto atMat2 = impl::aten::buildATen(mat2);

    bool is_sparse_input = false;
    bool is_sparse_mat2 = false;
    diopiIsTensorSparse(input, &is_sparse_input);
    diopiIsTensorSparse(mat2, &is_sparse_mat2);

    if (is_sparse_input && !is_sparse_mat2) {
        diopiConstTensorHandle_t crow_indices;
        diopiGetTensorCrowIndices(input, &crow_indices);

        diopiConstTensorHandle_t col_indices;
        diopiGetTensorColIndices(input, &col_indices);

        diopiConstTensorHandle_t values;
        diopiGetTensorValues(input, &values);

        auto atRowPtr = impl::aten::buildATen(crow_indices);
        auto atColInd = impl::aten::buildATen(col_indices);
        auto atValue = impl::aten::buildATen(values);
        auto atOut = impl::aten::buildATen(out);
        atOut.zero_();
        sparse::ops::row_balance_row_major_seq_reduce_kernel(atOut, atRowPtr, atColInd, atValue, atMat2);
        return diopiSuccess;

    } else if (is_sparse_input && is_sparse_mat2) {
        return diopiNoImplement;
    } else if (!is_sparse_input && !is_sparse_mat2) {
        return diopiNoImplement;
    } else {
        return diopiNoImplement;
    }

    return diopiErrorOccurred;
}


extern "C" diopiError_t diopiSpConv(diopiContextHandle_t ctx, diopiTensorHandle_t out_feat, diopiTensorHandle_t in_feat,
        diopicTensorHandle_t kernel, diopiTensorHandle_t neighbor_map,const int sum_nnz, 
        diopiTensorHandle_t neighbor_address, diopiTensorHandle_t q_neighbor_address, const int output_size, 
        const int qsum_nnz, const bool transpose, const bool allow_tf32, const bool allow_fp16 ) {
    
    impl::aten::setCurStream(ctx);

    auto atIn_feat = impl::aten::buildATen(in_feat);
    auto atOut_feat = impl::aten::buildATen(out_feat);
    auto atKernel = impl::aten::buildATen(kernel);
    auto atNeighbor_map = impl::aten::buildATen(neighbor_map);
    auto atNeighbor_address = impl::aten::buildATen(neighbor_address);
    auto atQ_neighbor_address = impl::aten::buildATen(q_neighbor_address);
    atOut_feat.zero_();

    sparse::ops::conv_forward_fetch_on_demand_cuda(atIn_feat, atOut_feat, atKernel, atNeighbor_map, sum_nnz, 
            atNeighbor_address,atQ_neighbor_address,output_size,
            qsum_nnz,transpose,allow_tf32,allow_fp16);
            
    return diopiSuccess;

}

}  // namespace cuda
}  // namespace impl
