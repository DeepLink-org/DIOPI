/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    AscendTensor outTensor(out);
    AclOpRunner<2, 1>("Pow", ctx).addInput(input, outTensor.dtype()).addInput(exponent, outTensor.dtype()).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    return diopiPowTensor(ctx, input, input, exponent);
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    AscendTensor outTensor(out);
    AclOpRunner<2, 1>("Pow", ctx).addInput(input, outTensor.dtype()).addConstInput(*exponent, outTensor.dtype()).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) { return diopiPow(ctx, input, input, exponent); }

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    AscendTensor outTensor(out);
    AclOpRunner<2, 1>("Pow", ctx).addConstInput(*input, outTensor.dtype()).addInput(exponent, outTensor.dtype()).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
