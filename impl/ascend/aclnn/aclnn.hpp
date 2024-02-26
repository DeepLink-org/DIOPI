/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACLNN_HPP_
#define IMPL_ASCEND_ACLNN_ACLNN_HPP_

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ascend_tensor.hpp"
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"  // TODO(zmz): add all
#include "aclnnop/aclnn_cos.h"  // TODO(zmz): add all
#include "aclnnop/aclnn_flash_attention_score.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"
#include "aclnnop/aclnn_sin.h"  // TODO(zmz): add all
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int aclnnAddAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiConstTensorHandle_t other, const diopiScalar_t* alpha,
                    diopiTensorHandle_t out);

int aclnnSinAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiTensorHandle_t out);

int aclnnCosAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiTensorHandle_t out);

int aclnnFlashAttentionAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t* softmaxMax, diopiTensorHandle_t* softmaxSum,
                               diopiTensorHandle_t* softmaxOut, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                               diopiConstTensorHandle_t v, double pDropout, double softmaxScale, bool isCausal);

int aclnnFlashAttentionBackwardAdaptor(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                       diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v,
                                       diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxMax, diopiConstTensorHandle_t softmaxSum,
                                       diopiConstTensorHandle_t softmaxOut, diopiGeneratorHandle_t gen, double pDropout, double softmaxScale, bool isCausal);
}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_ACLNN_ACLNN_HPP_

