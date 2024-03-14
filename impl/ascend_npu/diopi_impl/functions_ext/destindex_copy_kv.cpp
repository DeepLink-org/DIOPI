/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiDestIndexCopyKV(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t k, diopiConstTensorHandle_t destLoc) {
    BEGIN_CALL_ACL_OP(out, k, destLoc);
    auto orig_shape = destLocAt.sizes();
    if (destLocAt.sizes().size() != 1) {
        set_last_error_string("only support destLoc.rank == 1");
        return diopiNoImplement;
    }
    std::vector<int64_t> shape(destLocAt.dim() + 1, 1);
    for (int64_t i = 0; i < destLocAt.dim(); i++) {
        shape[i] = destLocAt.size(i);
    }
    auto destLocReshapeAt = impl::aten::viewStorage(destLocAt, shape);
    EXEC_NPU_CMD(aclnnScatterNd, outAt, destLocReshapeAt, kAt, outAt);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
