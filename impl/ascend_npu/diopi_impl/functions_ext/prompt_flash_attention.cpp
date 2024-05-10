/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"


namespace OP_IMPL_NS {

diopiError_t diopiPromptFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t query, diopiConstTensorHandle_t key,
                                       diopiConstTensorHandle_t value, diopiConstTensorHandle_t paddingMask, diopiConstTensorHandle_t attenMask,
                                       diopiSize_t actualSeqLengths, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
                                       const char* inputLayout, int64_t numKeyValueHeads) {
    BEGIN_CALL_ACL_OP(out, query, key, value, paddingMask, attenMask);
    at::IntArrayRef actSeqLen(actualSeqLengths.data, actualSeqLengths.len);
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnPromptFlashAttention, queryAt, keyAt, valueAt, paddingMaskAt,
        attenMaskAt, actSeqLen, numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, outAt);
    
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
