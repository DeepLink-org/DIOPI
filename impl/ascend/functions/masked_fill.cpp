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
    diopiTensorHandle_t valueTemp;
    if (typeSet.find(valueDtype) == typeSet.end()) {
        diopiSize_t valueSize;
        diopiGetTensorShape(value, &valueSize);
        diopiRequireTensor(ctx, &valueTemp, &valueSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, valueTemp, value);
    } else {
        valueTemp = (diopiTensorHandle_t)value;
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        diopiSize_t tensorSize;
        diopiGetTensorShape(input, &tensorSize);
        diopiRequireTensor(ctx, &inputTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, inputTemp, input);
        diopiGetTensorShape(out, &tensorSize);
        diopiRequireTensor(ctx, &outTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, outTemp, out);
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addInput(valueTemp).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addInput(valueTemp).addOutput(out).run();
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    return diopiMaskedFill(ctx, input, input, mask, value);
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        return diopiSuccess;
    }

    diopiTensorHandle_t scalarTensor;
    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_int32, diopi_dtype_int64};

    if (typeSet.find(value->stype) == typeSet.end()) {
        makeTensorFromScalar(ctx, value, &scalarTensor, diopi_dtype_float32);
    } else {
        makeTensorFromScalar(ctx, value, &scalarTensor, value->stype);
    }

    diopiDtype_t inputDtype;
    diopiGetTensorDtype(input, &inputDtype);
    if (typeSet.find(inputDtype) == typeSet.end()) {
        diopiTensorHandle_t inputTemp, outTemp;
        diopiSize_t tensorSize;
        diopiGetTensorShape(input, &tensorSize);
        diopiRequireTensor(ctx, &inputTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, inputTemp, input);
        diopiGetTensorShape(out, &tensorSize);
        diopiRequireTensor(ctx, &outTemp, &tensorSize, nullptr, diopi_dtype_float32, diopi_device);
        diopiCastDtype(ctx, outTemp, out);
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(inputTemp).addInput(mask).addConstInput(scalarTensor, ACL_FORMAT_ND, true).addOutput(outTemp).run();
        diopiCastDtype(ctx, out, outTemp);
    } else {
        AclOpRunner<3, 1>("MaskedFill", ctx).addInput(input).addInput(mask).addConstInput(scalarTensor, ACL_FORMAT_ND, true).addOutput(out).run();
    }

    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    return diopiMaskedFillScalar(ctx, input, input, mask, value);
}
}  // extern "C"

}  // namespace ascend
}  // namespace impl
