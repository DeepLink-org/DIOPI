/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(input);
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? 0.0 : 0,
                                        DiopiDataType::isFloatPoint(other->stype) ? 1.0 : 1,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    }
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? 0.0 : 0,
                                        DiopiDataType::isFloatPoint(other->stype) ? 1.0 : 1,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
