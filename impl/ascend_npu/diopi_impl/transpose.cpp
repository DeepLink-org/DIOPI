/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    BEGIN_CALL_ACL_OP(input, out);
    if (0 == inputAt.dim()) {
        outAt.copy_(inputAt);
        return diopiSuccess;
    }

    std::vector<int64_t> dims;
    dims.reserve(32);
    for (int i = 0; i < inputAt.dim(); i++) {
        dims.push_back(i);
    }
    if (dim0 < 0) {
        dim0 = dim0 + inputAt.dim();
    }
    if (dim1 < 0) {
        dim1 = dim1 + inputAt.dim();
    }
    dims[dim0] = dim1;
    dims[dim1] = dim0;
    at::IntArrayRef dimsAt(dims);
    EXEC_NPU_CMD(aclnnPermute, inputAt, dimsAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    BEGIN_CALL_ACL_OP(input, dims, out);
    if (0 == dims.len) {
        outAt.copy_(inputAt);
        return diopiSuccess;
    }
    EXEC_NPU_CMD(aclnnPermute, inputAt, dimsAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
