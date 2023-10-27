/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    AscendTensor outAt(out), inAt(input), t1(tensor1), t2(tensor2);
    if (inAt.numel() == 0) {
        return diopiSuccess;
    }
    auto size = inferSize(t1.shape(), t2.shape());
    size = inferSize(size, inAt.shape());
    broadcast(ctx, inAt, inAt, size);
    broadcast(ctx, t1, t1, size);
    broadcast(ctx, t2, t2, size);
    AclOpRunner<4, 1>("Addcdiv", ctx).addInput(inAt).addInput(t1).addInput(t2).addConstInput(*value, outAt.dtype()).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    return diopiAddcdiv(ctx, input, input, tensor1, tensor2, value);
}

}  // namespace ascend
}  // namespace impl
