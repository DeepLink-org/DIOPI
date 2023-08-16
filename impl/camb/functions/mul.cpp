/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

extern "C" diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(input);
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

extern "C" diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(input);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DIOPI_CALL(cnnlOpTensor(ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_MUL));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
