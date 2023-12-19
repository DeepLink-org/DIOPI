/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    AscendTensor asInput(input);
    AscendTensor asCond(condition);
    AscendTensor asOther(other);
    AscendTensor asOut(out);

    if (!(asInput.defined() && asCond.defined() && asOther.defined())) {
        return diopiSuccess;
    }

    if (asCond.dtype() != diopi_dtype_bool) {
        castTensor(ctx, asCond, diopi_dtype_bool);
    }

    broadcast(ctx, asInput, asInput, asOut.shape());
    broadcast(ctx, asCond, asCond, asOut.shape());
    broadcast(ctx, asOther, asOther, asOut.shape());

    AclOpRunner<3, 1>("Select", ctx).addInput(asCond).addInput(asInput).addInput(asOther).addOutput(out).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
