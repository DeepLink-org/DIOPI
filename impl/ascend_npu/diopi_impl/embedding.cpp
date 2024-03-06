/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    BEGIN_CALL_ACL_OP(weight, indices, out);
    EXEC_NPU_CMD(aclnnEmbedding, weightAt, indicesAt, outAt);
    END_CALL_ACL_OP();
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t numWeights, int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    BEGIN_CALL_ACL_OP(grad, indices, out);
    EXEC_NPU_CMD(aclnnEmbeddingDenseBackward, gradAt, indicesAt, numWeights, paddingIdx, scaleGradByfreq, outAt);
    END_CALL_ACL_OP();
}
}  // namespace OP_IMPL_NS
