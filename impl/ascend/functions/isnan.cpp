/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("IsNan", ctx).addInput(input, diopi_dtype_float64).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
