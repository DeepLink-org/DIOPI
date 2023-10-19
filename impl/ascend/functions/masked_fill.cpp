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

    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_int32, diopi_dtype_int64};

    diopiDtype_t valueDtype;
    diopiGetTensorDtype(value, &valueDtype);
    diopiTensorHandle_t valueTemp = (diopiTensorHandle_t)value;
    if (typeSet.find(valueDtype) == typeSet.end()) {
        makeTensorLike(ctx, &valueTemp, value, diopi_dtype_float32);
        diopiCastDtype(ctx, valueTemp, value);
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, outTemp, out);
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addInput(valueTemp).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addInput(valueTemp).addOutput(out).run();
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
    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_int32, diopi_dtype_int64};
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, outTemp, out);

        diopiTensorHandle_t scalarTensor;
        makeTensorFromScalar(ctx, value, &scalarTensor, diopi_dtype_float32, diopi_host);
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addConstInput(scalarTensor, ACL_FORMAT_ND, true).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        diopiTensorHandle_t scalarTensor;
        if (diopi_dtype_float16 != inputDtype) {
            makeTensorFromScalar(ctx, value, &scalarTensor, inputDtype, diopi_host);
        } else {
            makeTensorFromScalar(ctx, value, &scalarTensor, diopi_dtype_float32, diopi_host);
        }
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addConstInput(scalarTensor, ACL_FORMAT_ND, true).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    return diopiMaskedFillScalar(ctx, input, input, mask, value);
}

}  // namespace ascend
}  // namespace impl
