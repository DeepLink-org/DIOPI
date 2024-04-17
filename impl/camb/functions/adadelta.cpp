#include "../common/common.hpp"
#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t param, diopiConstTensorHandle_t grad, diopiTensorHandle_t squareAvg,
                           diopiTensorHandle_t accDelta, float lr, float rho, float eps, float weightDecay) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor paramTenso = DiopiTensor(param);
    DiopiTensor gradTensor = DiopiTensor(grad);
    DiopiTensor squareAvgTensor = DiopiTensor(squareAvg);
    DiopiTensor accDeltaTensor = DiopiTensor(accDelta);

    DiopiTensor paramCasted = paramTensor;
    DiopiTensor squareAvgCasted = squareAvgTensor;
    DiopiTensor accDeltaCasted = accDeltaTensor;

    DiopiTensor gradCasted;
    DIOPI_CALL(clone(ctx, gradTensor, gradCasted));

    std::vector<DiopiTensor*> tensors{&paramCasted, &gradCasted, &squareAvgCasted, &accDeltaCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    if (weightDecay != 0.0f) {
        DIOPI_CALL(cnnlOpTensor(ctx, paramCasted, gradCasted, gradCasted, CNNL_OP_TENSOR_ADD, static_cast<double>(weightDecay), 1.0, 0.0));
    }

    CnnlTensorDesc paramDesc(paramCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc squareAvgDesc(squareAvgCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc accDeltaDesc(accDeltaCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradCasted, CNNL_LAYOUT_ARRAY);

    DiopiTensor lrTensor, rhoTensor, epsTensor;
    diopiScalar_t lrScalar = constructDiopiScalarT(diopi_dtype_float64, lr);
    diopiScalar_t rhoScalar = constructDiopiScalarT(diopi_dtype_float64, rho);
    diopiScalar_t epsScalar = constructDiopiScalarT(diopi_dtype_float64, eps);

    DIOPI_CALL(makeTensorFromScalar(ctx, &lrScalar, lrTensor));
    DIOPI_CALL(makeTensorFromScalar(ctx, &rhoScalar, rhoTensor));
    DIOPI_CALL(makeTensorFromScalar(ctx, &epsScalar, epsTensor));

    DIOPI_CALL(dataTypeCast(ctx, lrTensor, paramCasted.dtype()));
    DIOPI_CALL(dataTypeCast(ctx, rhoTensor, paramCasted.dtype()));
    DIOPI_CALL(dataTypeCast(ctx, epsTensor, paramCasted.dtype()));

    DIOPI_CALL_CNNL(cnnlApplyAdadelta(handle,
                                      paramDesc.get(),
                                      paramCasted.data(),
                                      squareAvgDesc.get(),
                                      squareAvgCasted.data(),
                                      accDeltaDesc.get(),
                                      accDeltaCasted.data(),
                                      gradDesc.get(),
                                      gradCasted.data(),
                                      lrTensor.data(),
                                      rhoTensor.data(),
                                      epsTensor.data()));

    DIOPI_CALL(dataTypeCast(ctx, paramTensor, paramCasted));
    DIOPI_CALL(dataTypeCast(ctx, squareAvgTensor, squareAvgCasted));
    DIOPI_CALL(dataTypeCast(ctx, accDeltaTensor, accDeltaCasted));
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
