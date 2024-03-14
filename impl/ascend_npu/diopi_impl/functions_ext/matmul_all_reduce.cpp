/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiMatmulAllReduce(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x1,
                                  diopiConstTensorHandle_t x2, diopiConstTensorHandle_t bias, const char* group,
                                  const char* reduceOp, int64_t commTurn, int64_t streamMode) {
    BEGIN_CALL_ACL_OP(out, x1, x2, bias);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnMatmulAllReduce, x1At, x2At, biasAt, group, reduceOp, commTurn, streamMode, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS