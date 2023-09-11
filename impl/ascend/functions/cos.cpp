/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiCos(ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64, diopi_dtype_complex64, diopi_dtype_complex128};

    // only support: float16, float32, int32, int64, double, complex64, complex128.
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        diopiSize_t tensorSize;
        diopiGetTensorShape(input, &tensorSize);
        diopiRequireTensor(ctx, &inputTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiRequireTensor(ctx, &outTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, inputTemp, input);
        AclOpRunner<1, 1>("Cos", ctx).addInput(inputTemp).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<1, 1>("Cos", ctx).addInput(input).addOutput(out).run();
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
