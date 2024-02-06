/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                             diopiConstTensorHandle_t value) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    std::set<diopiDtype_t> typeSet{diopi_dtype_bool, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_int32, diopi_dtype_int64};

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, valutTemp, outTemp;
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
        makeTensorLike(ctx, &valutTemp, value, diopi_dtype_float32);
        diopiCastDtype(ctx, valutTemp, value);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, outTemp, out);

        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addInput(valutTemp).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addInput(value, inputDtype).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    return diopiMaskedFill(ctx, input, input, mask, value);
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    std::set<diopiDtype_t> typeSet{diopi_dtype_bool, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_int32, diopi_dtype_int64};
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, outTemp, out);

        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addConstInput(*value, diopi_dtype_float32).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addConstInput(*value, inputDtype).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    return diopiMaskedFillScalar(ctx, input, input, mask, value);
}

}  // namespace ascend
}  // namespace impl
