/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    std::set<diopiDtype_t> typeSet{
        diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32, diopi_dtype_int64, diopi_dtype_float64, diopi_dtype_complex64, diopi_dtype_complex128};

    // only support: float16, float32, int32, int64, double, complex64, complex128.
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        diopiSize_t tensorSize;
        diopiGetTensorShape(input, &tensorSize);
        diopiRequireTensor(ctx, &inputTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiRequireTensor(ctx, &outTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, inputTemp, input);
        AclOpRunner<1, 1>("Sign", ctx).addInput(inputTemp).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<1, 1>("Sign", ctx).addInput(input).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) { return diopiSgn(ctx, input, input); }

}  // namespace ascend
}  // namespace impl
