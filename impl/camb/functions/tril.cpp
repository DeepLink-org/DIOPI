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

extern "C" {

DIOPI_API diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, {diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlTri(handle, static_cast<int>(diagonal), false, inputDesc.get(), inputTensorTmp.data(), outDesc.get(), outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
