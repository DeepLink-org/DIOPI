/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    AclOpRunner<2, 1>("ArgMaxV2", ctx).addInput(input).addConstInput(*dim, diopi_dtype_int64).setAttr("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
