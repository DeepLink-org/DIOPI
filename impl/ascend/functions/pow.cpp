/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    AscendTensor inputAt(input), expAt(exponent), outAt(out);
    if (inputAt.numel() == 0 || expAt.numel() == 0) {
        return diopiSuccess;
    }
    auto dtype = promoteTypes(inputAt.dtype(), expAt.dtype());
    castTensor(ctx, inputAt, dtype);
    castTensor(ctx, expAt, dtype);
    AclOpRunner<2, 1>("Pow", ctx).addInput(inputAt).addInput(expAt).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    return diopiPowTensor(ctx, input, input, exponent);
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiTensorHandle_t exponentTensor;
    AscendTensor inputAt(input);

    if ((isFloatingType(inputAt.dtype()) && isFloatingType(exponent->stype)) ||
        ((isIntegralTypeWithBool(inputAt.dtype()) && isIntegralTypeWithBool(inputAt.dtype())))) {
        makeTensorFromScalar(ctx, exponent, &exponentTensor, inputAt.dtype(), diopi_device);
    } else if (isIntegralTypeWithBool(inputAt.dtype()) && isFloatingType(exponent->stype)) {
        makeTensorFromScalar(ctx, exponent, &exponentTensor, diopi_dtype_float32, diopi_device);
    } else if (isFloatingType(inputAt.dtype()) && isIntegralTypeWithBool(inputAt.dtype())) {
        makeTensorFromScalar(ctx, exponent, &exponentTensor, inputAt.dtype(), diopi_device);
    }

    return diopiPowTensor(ctx, out, input, exponentTensor);
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) { return diopiPow(ctx, input, input, exponent); }

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    diopiTensorHandle_t inputTensor;
    AscendTensor expAt(exponent);

    if ((isFloatingType(input->stype) && isFloatingType(expAt.dtype())) || ((isIntegralTypeWithBool(input->stype) && isIntegralTypeWithBool(expAt.dtype())))) {
        makeTensorFromScalar(ctx, input, &inputTensor, expAt.dtype(), diopi_device);
    } else if (isIntegralTypeWithBool(input->stype) && isFloatingType(expAt.dtype())) {
        makeTensorFromScalar(ctx, input, &inputTensor, expAt.dtype(), diopi_device);
    } else if (isFloatingType(input->stype) && isIntegralTypeWithBool(expAt.dtype())) {
        makeTensorFromScalar(ctx, input, &inputTensor, diopi_dtype_float32, diopi_device);
    }

    return diopiPowTensor(ctx, out, inputTensor, exponent);
}

}  // namespace ascend
}  // namespace impl
