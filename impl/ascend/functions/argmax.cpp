/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    AscendTensor inputAt(input);
    int64_t dimValue;
    if (nullptr == dim) {
        inputAt.view({inputAt.numel()});
        dimValue = 0;
        keepdim = false;
    } else {
        dimValue = *dim;
    }
    AclOpRunner<2, 1>("ArgMaxV2", ctx).addInput(inputAt).addConstInput(dimValue, diopi_dtype_int64).setAttr("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
