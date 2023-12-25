/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size,
                         diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    BEGIN_CALL_ACL_OP(input, output_size, kernel_size, dilation, padding, stride, out);
    acl_op::col2im_out(inputAt, output_sizeAt, kernel_sizeAt, dilationAt, paddingAt, strideAt, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
