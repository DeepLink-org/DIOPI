/**
 * @file ones.cpp
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiZeros(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiSize_t size) {
    BEGIN_CALL_ACL_OP(out);
    std::vector<int64_t> sizeVec(size.data, size.data + size.len);
    op_api::zeros_out(sizeVec, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiZeroInp(diopiContextHandle_t ctx, diopiTensorHandle_t self) {
    BEGIN_CALL_ACL_OP(self);
    EXEC_NPU_CMD(aclnnInplaceZero, self);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
