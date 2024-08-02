/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include <numeric>

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, const diopiScalar_t* correction) {
    BEGIN_CALL_ACL_OP(out, input, dim, correction);
    if (correction == nullptr) {
        correctionAt = 1;  // default correction value in torch_std is 1
    }
    bool keepdim = false;
    if (inputAt.dim() == outAt.dim()) {
        keepdim = true;
    }
    if (0 == dim.len) {
        c10::DimVector adim(inputAt.dim());
        std::iota(adim.begin(), adim.end(), 0);
        at::IntArrayRef rdim(adim.data(), adim.size());
        op_api::std_out(inputAt, rdim, correctionAt, keepdim, outAt);
    } else {
        at::IntArrayRef rdim(dim.data, dim.len);
        op_api::std_out(inputAt, rdim, correctionAt, keepdim, outAt);
    }
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
