/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outputTensor(out);
    bool isInputContiguous = inputTensor.isContiguous();
    bool isOtherContiguous = otherTensor.isContiguous();
    if (isInputContiguous && (!isOtherContiguous)) {
        DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::Contiguous));
    } else if ((!isInputContiguous) && isOtherContiguous) {
        if (inputTensor.dim() == 5) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast3d));
        } else if (inputTensor.dim() == 4) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast));
        } else if (inputTensor.dim() == 3) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast1d));
        } else if (inputTensor.dim() == 2) {
            DIOPI_CALL(contiguous(ctx, inputTensor, diopiMemoryFormat_t::Contiguous));
        }
    }
    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    bool isInputContiguous = inputTensor.isContiguous();
    bool isOtherContiguous = otherTensor.isContiguous();
    if (isInputContiguous && (!isOtherContiguous)) {
        DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::Contiguous));
    } else if ((!isInputContiguous) && isOtherContiguous) {
        if (inputTensor.dim() == 5) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast3d));
        } else if (inputTensor.dim() == 4) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast));
        } else if (inputTensor.dim() == 3) {
            DIOPI_CALL(contiguous(ctx, otherTensor, diopiMemoryFormat_t::ChannelsLast1d));
        } else if (inputTensor.dim() == 2) {
            DIOPI_CALL(contiguous(ctx, inputTensor, diopiMemoryFormat_t::Contiguous));
        }
    }
    DiopiTensor outputTensor(input);

    DIOPI_CALL(cnnlOpTensor(
        ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype) && DiopiDataType::isInteger(alpha->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival,
                                        DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival,
                                        DiopiDataType::isFloatPoint(inputTensor.dtype()) ? 1.0 : 1));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(
            ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    }
    return diopiSuccess;
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float16 || inputTensor.dtype() == diopi_dtype_float32 ||
        (inputTensor.dtype() == diopi_dtype_int32 && DiopiDataType::isInteger(other->stype) && DiopiDataType::isInteger(alpha->stype))) {
        DIOPI_CALL(cnnlTransformAdaptor(ctx,
                                        outputTensor,
                                        inputTensor,
                                        DiopiDataType::isFloatPoint(other->stype) ? other->fval : other->ival,
                                        DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival,
                                        DiopiDataType::isFloatPoint(inputTensor.dtype()) ? 1.0 : 1));
    } else {
        DiopiTensor otherTensor;
        DIOPI_CALL(makeTensorFromScalar(ctx, other, otherTensor));
        DIOPI_CALL(cnnlOpTensor(
            ctx, inputTensor, otherTensor, outputTensor, CNNL_OP_TENSOR_ADD, 1.0, DiopiDataType::isFloatPoint(alpha->stype) ? alpha->fval : alpha->ival));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
