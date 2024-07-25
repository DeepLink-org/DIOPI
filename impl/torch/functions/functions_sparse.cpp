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
        diopiTensorHandle_t crow_indices;
        diopiGetTensorCrowIndices(input, &crow_indices);

        diopiTensorHandle_t col_indices;
        diopiGetTensorColIndices(input, &col_indices);

        diopiTensorHandle_t values;
        diopiGetTensorValues(input, &values);

        auto atRowPtr = impl::aten::buildATen(crow_indices);
        auto atColInd = impl::aten::buildATen(col_indices);
        auto atValue = impl::aten::buildATen(values);
        auto atOut = impl::aten::buildATen(out);
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

}  // namespace cuda
}  // namespace impl
