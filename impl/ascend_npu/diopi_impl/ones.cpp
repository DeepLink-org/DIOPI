/**
 * @file ones.cpp
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiOnes(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiSize_t size) {
    BEGIN_CALL_ACL_OP(out);
    c10::DimVector sizeVec(size.data, size.data + size.len);
    op_api::ones_out(sizeVec, outAt);
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
