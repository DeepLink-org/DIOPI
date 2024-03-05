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

static const int64_t MIN_DEPTH = 1;
static const int64_t AUTO_DEPTH = -1;
static const int64_t MIN_NUM_CLASSES = 0;

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    BEGIN_CALL_ACL_OP(input, out);
    int64_t depth = numClasses;
    TORCH_CHECK(depth >= AUTO_DEPTH, "NPU error, not yet support negative num_classes, when num_classes less than -1");
    TORCH_CHECK(inputAt.numel() != 0 || numClasses > MIN_NUM_CLASSES, "NPU error, can not infer total number of classes from empty tensor.");
    if (depth == AUTO_DEPTH) {
        depth = inputAt.max().item().toLong() + 1;
        if (depth < MIN_DEPTH) {
            depth = MIN_DEPTH;
        }
    }
    at::Tensor on_value_tensor = at_npu::native::OpPreparation::apply_tensor_without_format({1}, inputAt.options());
    op_api::fill_(on_value_tensor, 1);
    at::Tensor off_value_tensor = at_npu::native::OpPreparation::apply_tensor_without_format({1}, inputAt.options());
    op_api::fill_(off_value_tensor, 0);
    auto output_size = op_infer::array_to_small_vector(inputAt.sizes());
    output_size.emplace_back(depth);
    int64_t axis = -1;
    EXEC_NPU_CMD(aclnnOneHot, inputAt, depth, on_value_tensor, off_value_tensor, axis, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
