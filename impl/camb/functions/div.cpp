/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

DIOPI_API diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                                diopiRoundMode_t roundingMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);
    DIOPI_CALL(diopiDivInternal(ctx, inputTensor, otherTensor, outTensor, roundingMode));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t roundingMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(input);
    DIOPI_CALL(diopiDivInternal(ctx, inputTensor, otherTensor, outTensor, roundingMode));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                                      diopiRoundMode_t roundingMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DiopiTensor outTensor(out);
    DIOPI_CALL(diopiDivInternal(ctx, inputTensor, otherTensor, outTensor, roundingMode));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t roundingMode) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
    DiopiTensor outTensor(input);
    DIOPI_CALL(diopiDivInternal(ctx, inputTensor, otherTensor, outTensor, roundingMode));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
