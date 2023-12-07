/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

static diopiError_t indexFill(diopiContextHandle_t ctx, const int dim, float value, DiopiTensor input, DiopiTensor index, DiopiTensor& output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor outputTmp = output;
    if (input.dtype() != output.dtype()) {
        outputTmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputTmpDesc(outputTmp, CNNL_LAYOUT_ARRAY);
    DIOPI_CALL_CNNL(cnnlIndexFill(handle, dim, value, inputDesc.get(), input.data(), indexDesc.get(), index.data(), outputTmpDesc.get(), outputTmp.data()));
    if (outputTmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, outputTmp));
    }
    return diopiSuccess;
}

static diopiError_t indexFill(diopiContextHandle_t ctx, const int dim, DiopiTensor value, DiopiTensor input, DiopiTensor index, DiopiTensor& output) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor outputTmp = output;
    if (input.dtype() != output.dtype()) {
        outputTmp = requiresTensor(ctx, output.shape(), input.dtype());
    }
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputTmpDesc(outputTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL(dataTypeCast(ctx, value, input.dtype()));

    DIOPI_CALL_CNNL(cnnlIndexFill_v2(handle,
                                     dim,
                                     CNNL_POINTER_MODE_DEVICE,
                                     value.data(),
                                     inputDesc.get(),
                                     input.data(),
                                     indexDesc.get(),
                                     index.data(),
                                     outputTmpDesc.get(),
                                     outputTmp.data()));
    if (outputTmp.dtype() != output.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output, outputTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                            diopiConstTensorHandle_t value) {
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor indexTensor(index);
    DiopiTensor valueTensor(value);
    DIOPI_CALL(indexFill(ctx, dim, valueTensor, inputTensor, indexTensor, outTensor));
    return diopiSuccess;
}

diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                               diopiConstTensorHandle_t value) {
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    DiopiTensor valueTensor(value);
    DIOPI_CALL(indexFill(ctx, dim, valueTensor, inputTensor, indexTensor, inputTensor));
    return diopiSuccess;
}

diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                  diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DiopiTensor indexTensor(index);
    float scalarValue = DiopiDataType::isFloatPoint(value->stype) ? value->fval : value->ival;
    DIOPI_CALL(indexFill(ctx, dim, scalarValue, inputTensor, indexTensor, outTensor));
    return diopiSuccess;
}

diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                                     const diopiScalar_t* value) {
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    float scalarValue = DiopiDataType::isFloatPoint(value->stype) ? value->fval : value->ival;
    DIOPI_CALL(indexFill(ctx, dim, scalarValue, inputTensor, indexTensor, inputTensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
