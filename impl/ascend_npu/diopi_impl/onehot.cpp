/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

static const int64_t minDepth = 1;
static const int64_t authDepth = -1;
static const int64_t minNumClass = 0;

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    BEGIN_CALL_ACL_OP(input, out);
    int64_t depth = numClasses;
    TORCH_CHECK(depth >= authDepth, "NPU error, not yet support negative num_classes, when num_classes less than -1");
    TORCH_CHECK(inputAt.numel() != 0 || numClasses > minNumClass, "NPU error, can not infer total number of classes from empty tensor.");
    if (depth == authDepth) {
        depth = op_api::max(inputAt).item().toLong() + 1;
        if (depth < minDepth) {
            depth = minDepth;
        }
    }
    at::Tensor onValueTensor = at_npu::native::OpPreparation::apply_tensor_without_format({1}, inputAt.options());
    op_api::fill_(onValueTensor, 1);
    at::Tensor offValueTensor = at_npu::native::OpPreparation::apply_tensor_without_format({1}, inputAt.options());
    op_api::fill_(offValueTensor, 0);
    int64_t axis = -1;
    EXEC_NPU_CMD(aclnnOneHot, inputAt, depth, onValueTensor, offValueTensor, axis, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
