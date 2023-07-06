#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
namespace {

int getDim(DiopiTensor tensor, int64_t dim) {
    int shapeSize = tensor.shape().size();
    int dimTmp = static_cast<int>(dim);
    if (dimTmp < 0) {
        dimTmp = dim + shapeSize;
    }
    return dimTmp;
}

}  // namespace

extern "C" {

diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensor}, {diopi_dtype_int32, diopi_dtype_float32, diopi_dtype_float16}));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    int axis = getDim(inputTensor, dim);

    if (inputTensor.dtype() == outTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlCumsum(handle, inputDesc.get(), inputTensor.data(), axis, false, false, CNNL_PROPAGATE_NAN, outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTemp = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCumsum(handle, inputDesc.get(), inputTensor.data(), axis, false, false, CNNL_PROPAGATE_NAN, outTempDesc.get(), outTemp.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
