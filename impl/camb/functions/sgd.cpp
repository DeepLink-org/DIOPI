/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {

// a= a * scale_a + b * scale_b;
static diopiError_t addMulFunc(diopiContextHandle_t ctx, DiopiTensor &a, float scaleA, const DiopiTensor &b, float scaleB) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    size_t workspaceSize;
    std::vector<int> shape;
    shape.push_back(a.numel());
    CnnlTensorDesc aDesc, bDesc;
    DIOPI_CALL(aDesc.set(a, CNNL_LAYOUT_ARRAY, shape));
    DIOPI_CALL(bDesc.set(b, CNNL_LAYOUT_ARRAY, shape));

    DIOPI_CALL_CNNL(cnnlGetBiasAddWorkspaceSize(handle, bDesc.get(), aDesc.get(), &workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALL_CNNL(cnnlBiasAdd(handle, &scaleB, bDesc.get(), b.data(), workspace, workspaceSize, &scaleA, aDesc.get(), a.data()));
    return diopiSuccess;
}

diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t param, diopiTensorHandle_t grad, diopiTensorHandle_t buf, double lr, double momentum,
                      double dampening, double weightDecay, bool nesterov) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor paramTensor(param);
    DiopiTensor gradTensor(grad);
    DiopiTensor bufTensor;
    DiopiTensor bufTensorTmp;
    diopiDtype_t gradDtypeOrigin = gradTensor.dtype();
    std::vector<DiopiTensor *> pTensors;
    if (buf != nullptr) {
        bufTensor = DiopiTensor(buf);
        bufTensorTmp = bufTensor;
        pTensors = std::vector<DiopiTensor *>{&gradTensor, &bufTensorTmp};
    } else {
        pTensors = std::vector<DiopiTensor *>{&gradTensor};
    }
    DiopiTensor gradTensorTmp = gradTensor;
    if (gradDtypeOrigin == gradTensor.dtype()) {
        DIOPI_CALL(clone(ctx, gradTensor, gradTensorTmp));
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor paramTensorTmp = paramTensor;
    if (gradTensorTmp.dtype() != paramTensorTmp.dtype()) {
        paramTensorTmp = requiresTensor(ctx, paramTensor.shape(), gradTensorTmp.dtype());
        DIOPI_CALL(dataTypeCast(ctx, paramTensorTmp, paramTensor));
    }

    CnnlTensorDesc pa ra m(paramTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradTensorTmp, CNNL_LAYOUT_ARRAY);

    if (weightDecay != 0) {
        DIOPI_CALL(addMulFunc(ctx, gradTensorTmp, 1.0, paramTensorTmp, weightDecay));
    }
    if (momentum != 0) {
        if (buf == nullptr) {
            bufTensorTmp = gradTensorTmp;
        } else {
            DIOPI_CALL(addMulFunc(ctx, bufTensorTmp, momentum, gradTensorTmp, (1.0 - dampening)));
        }
        if (nesterov) {
            DIOPI_CALL(addMulFunc(ctx, gradTensorTmp, 1.0, bufTensorTmp, momentum));
        } else {
            gradTensorTmp = bufTensorTmp;
        }
    }

    std::vector<int64_t> shape{1};
    DiopiTensor lrTensor;
    diopiScalar_t lrScalar = constructDiopiScalarT(diopi_dtype_float64, lr);
    DIOPI_CALL(makeTensorFromScalar(ctx, &lrScalar, lrTensor));
    DIOPI_CALL(dataTypeCast(ctx, lrTensor, gradTensorTmp.dtype()));
    DIOPI_CALL_CNNL(cnnlGradientDescent(handle, g ra dDesc.get(), gradTensorTmp.data(), lrTensor.data(), paramDesc.get(), paramTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, paramTensor, paramTensorTmp));
    if (buf != nullptr) {
        DIOPI_CALL(dataTypeCast(ctx, bufTensor, bufTensorTmp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
