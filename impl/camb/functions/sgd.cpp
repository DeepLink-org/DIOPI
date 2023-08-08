/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {

// out = a * scale_a + b * scale_b;
static diopiError_t addMulFunc(diopiContextHandle_t ctx, const DiopiTensor &a, float scaleA, const DiopiTensor &b, float scaleB, DiopiTensor &out) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    size_t workspaceSize;
    std::vector<int> shape;
    shape.push_back(a.numel());
    CnnlTensorDesc outDesc, bDesc;
    DIOPI_CALL(clone(ctx, a, out));
    DIOPI_CALL(outDesc.set(out, CNNL_LAYOUT_ARRAY, shape));
    DIOPI_CALL(bDesc.set(b, CNNL_LAYOUT_ARRAY, shape));

    DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, bDesc.get(), outDesc.get(), &workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlBiasAdd(handle, &scaleB, bDesc.get(), b.data(), workspace, workspaceSize, &scaleA, outDesc.get(), out.data()));
    return diopiSuccess;
};

extern "C" diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum,
                                 double dampening, double weightDecay, bool nesterov) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor wTensor(w);
    DiopiTensor dwTensor(dw);
    DiopiTensor bufTensor;
    DiopiTensor bufTensorTmp;
    diopiDtype_t dwDtypeOrign = dwTensor.dtype();
    std::vector<DiopiTensor *> pTensors;
    if (buf != nullptr) {
        bufTensor = DiopiTensor(buf);
        bufTensorTmp = bufTensor;
        pTensors = std::vector<DiopiTensor *>{&dwTensor, &bufTensorTmp};
    } else {
        pTensors = std::vector<DiopiTensor *>{&dwTensor};
    }
    DiopiTensor dwTensorTmp = dwTensor;
    if (dwDtypeOrign == dwTensor.dtype()) {
        DIOPI_CALL(clone(ctx, dwTensor, dwTensorTmp));
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor wTensorTmp = wTensor;
    if (dwTensorTmp.dtype() != wTensorTmp.dtype()) {
        wTensorTmp = requiresTensor(ctx, wTensor.shape(), dwTensorTmp.dtype());
        DIOPI_CALL(dataTypeCast(ctx, wTensorTmp, wTensor));
    }

    CnnlTensorDesc wDescTmp(wTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc dwDesc(dwTensorTmp, CNNL_LAYOUT_ARRAY);

    // a = a * scale_a + b * scale_b;
    auto addMulFunc = [&](auto &a, float scaleA, auto b, float scaleB) {
        size_t workspaceSize;
        std::vector<int> shape;
        shape.push_back(a.numel());
        CnnlTensorDesc aDesc, bDesc;
        DIOPI_CALL(aDesc.set(a, CNNL_LAYOUT_ARRAY, shape));
        DIOPI_CALL(bDesc.set(b, CNNL_LAYOUT_ARRAY, shape));

        DIOPI_CALLCNNL(cnnlGetBiasAddWorkspaceSize(handle, bDesc.get(), aDesc.get(), &workspaceSize));

        void *workspace = nullptr;
        if (workspaceSize != 0) {
            workspace = requiresBuffer(ctx, workspaceSize).data();
        }

        DIOPI_CALLCNNL(cnnlBiasAdd(handle, &scaleB, bDesc.get(), b.data(), workspace, workspaceSize, &scaleA, aDesc.get(), a.data()));
        return diopiSuccess;
    };

    if (weightDecay != 0) {
        DIOPI_CALL(addMulFunc(dwTensorTmp, 1.0, wTensorTmp, weightDecay));
    }
    if (momentum != 0) {
        if (buf == nullptr) {
            bufTensorTmp = dwTensorTmp;
        } else {
            DIOPI_CALL(addMulFunc(bufTensorTmp, momentum, dwTensorTmp, (1.0 - dampening)));
        }
        if (nesterov) {
            DIOPI_CALL(addMulFunc(dwTensorTmp, 1.0, bufTensorTmp, momentum));
        } else {
            dwTensorTmp = bufTensorTmp;
        }
    }

    std::vector<int64_t> shape{1};
    diopiSize_t size(shape.data(), shape.size());
    DiopiTensor lrTensor;
    diopiScalar_t lrScalar{diopi_dtype_float64, {lr}};
    DIOPI_CALL(makeTensorFromScalar(ctx, &lrScalar, lrTensor));
    DIOPI_CALL(dataTypeCast(ctx, lrTensor, dwTensorTmp.dtype()));
    DIOPI_CALLCNNL(cnnlGradientDescent(handle, dwDesc.get(), dwTensorTmp.data(), lrTensor.data(), wDescTmp.get(), wTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, wTensor, wTensorTmp));
    if (buf != nullptr) {
        DIOPI_CALL(dataTypeCast(ctx, bufTensor, bufTensorTmp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
