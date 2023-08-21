/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                 const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(input);
    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                       const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(input);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
