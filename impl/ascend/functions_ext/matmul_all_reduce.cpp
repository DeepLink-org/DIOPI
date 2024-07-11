/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMatmulAllReduce(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x1, diopiConstTensorHandle_t x2,
                                  diopiConstTensorHandle_t bias, const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmulAllReduce, ctx, x1, x2, bias, group, reduceOp, commTurn, streamMode, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
