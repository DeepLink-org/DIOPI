/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (diopi_dtype_bool == dtype) {
        AclOpRunner<1, 1>("LogicalNot", ctx).addInput(input).addOutput(out).run();
    } else {
        AclOpRunner<1, 1>("Invert", ctx).addInput(input).addOutput(out).run();
    }
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    if (diopi_dtype_bool == dtype) {
        AclOpRunner<2, 1>("LogicalAnd", ctx).addInput(input).addInput(other).addOutput(out).run();
    } else {
        AclOpRunner<2, 1>("BitwiseAnd", ctx).addInput(input).addInput(other).addOutput(out).run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
