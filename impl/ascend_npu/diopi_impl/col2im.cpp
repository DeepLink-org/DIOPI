/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize, diopiSize_t kernelSize,
                         diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    BEGIN_CALL_ACL_OP(input, outputSize, kernelSize, dilation, padding, stride, out);
#if 0
    acl_op::col2im_out(inputAt, outputSizeAt, kernelSizeAt, dilationAt, paddingAt, strideAt, outAt);
#endif
    op_api::col2im_out(inputAt, outputSizeAt, kernelSizeAt, dilationAt, paddingAt, strideAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
