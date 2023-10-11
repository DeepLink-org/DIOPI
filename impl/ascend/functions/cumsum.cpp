/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiDtype_t inputDtype;
    diopiDtype_t outputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(out, &outputDtype);

    if (inputDtype != outputDtype) {
        AscendTensor inputCopy(input);
        castTensor(ctx, inputCopy, outputDtype);
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(inputCopy).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    } else if (inputDtype == diopi_dtype_bool) {
        AscendTensor inputCopy(input);
        castTensor(ctx, inputCopy, outputDtype);
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(inputCopy).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    } else {
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(input).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
