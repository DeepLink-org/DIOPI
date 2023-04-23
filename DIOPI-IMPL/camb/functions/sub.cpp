/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <memory>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                 const diopiScalar_t* alpha) {
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor output_tensor(out);
    DIOPI_CALL(cnnl_op_tensor(
        ctx, input_tensor, other_tensor, output_tensor, CNNL_OP_TENSOR_SUB, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor output_tensor(input);
    DIOPI_CALL(cnnl_op_tensor(
        ctx, input_tensor, other_tensor, output_tensor, CNNL_OP_TENSOR_SUB, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                       const diopiScalar_t* alpha) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DiopiTensor other_tensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, other_tensor));
    DIOPI_CALL(cnnl_op_tensor(
        ctx, input_tensor, other_tensor, output_tensor, CNNL_OP_TENSOR_SUB, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(input);
    DiopiTensor other_tensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, other_tensor));
    DIOPI_CALL(cnnl_op_tensor(
        ctx, input_tensor, other_tensor, output_tensor, CNNL_OP_TENSOR_SUB, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
