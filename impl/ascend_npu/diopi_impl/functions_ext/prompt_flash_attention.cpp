/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiPromptFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t query, diopiConstTensorHandle_t key,
                                       diopiConstTensorHandle_t value, diopiConstTensorHandle_t paddingMask, diopiConstTensorHandle_t attenMask,
                                       diopiSize_t actualSeqLengths, int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
                                       const char* inputLayout, int64_t numKeyValueHeads) {
    BEGIN_CALL_ACL_OP(out, query, key, value, paddingMask, attenMask);
    at::IntArrayRef actSeqLen(actualSeqLengths.data, actualSeqLengths.len);
    c10::string_view atInputLayout(inputLayout, strlen(inputLayout));
    outAt.copy_(op_api::npu_prompt_flash_attention(
        queryAt, keyAt, valueAt, paddingMaskAt, attenMaskAt, actSeqLen, numHeads, scaleValue, preTokens, nextTokens, atInputLayout, numKeyValueHeads));
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
