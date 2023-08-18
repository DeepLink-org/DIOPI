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

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    DIOPI_CHECK(paddingIdx >= std::numeric_limits<int32_t>::min() && paddingIdx <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");
    int32_t paddingIdxCasted = static_cast<int32_t>(paddingIdx);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor weightTensor(weight);
    DiopiTensor indicesTensor(indices);

    DIOPI_CHECK(paddingIdx >= -1 && paddingIdx < weightTensor.shape().front(), "padding_idx should be valid");

    std::vector<DiopiTensor *> tensors{&indicesTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));

    DiopiTensor outTensorTmp = outTensor;
    // special case: the indices_tensor is empty
    if (indicesTensor.dim() == 0 && indicesTensor.numel() == 1) {
        outTensorTmp.unsqueeze(0);
    }
    if (weightTensor.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), weightTensor.dtype());
    }

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indicesDesc(indicesTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlEmbeddingForward_v2(handle,
                                           weightDesc.get(),
                                           weightTensor.data(),
                                           indicesDesc.get(),
                                           static_cast<const int *>(indicesTensor.data()),
                                           paddingIdxCasted,
                                           nullptr,
                                           nullptr,
                                           outDesc.get(),
                                           outTensorTmp.data()));
    if (outTensorTmp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t numWeights, int64_t paddingIdx, bool scaleGradByfreq, bool sparse) {
    DIOPI_CHECK(paddingIdx >= std::numeric_limits<int32_t>::min() && paddingIdx <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");
    DIOPI_CHECK(numWeights >= std::numeric_limits<int32_t>::min() && numWeights <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");

    int32_t paddingIdxCasted = static_cast<int32_t>(paddingIdx);
    int32_t numWeightsCasted = static_cast<int32_t>(numWeights);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor gradTensor(grad);
    DiopiTensor indicesTensor(indices);

    DIOPI_CHECK(outTensor.shape().front() == numWeightsCasted && outTensor.shape().back() == gradTensor.shape().back(), "mismatch of shape");

    std::vector<DiopiTensor *> tensors{&gradTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));

    std::vector<DiopiTensor *> tensors1{&indicesTensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors1, {diopi_dtype_int32}));

    DiopiTensor outTensorTmp = outTensor;
    // special case: the indices_tensor is empty
    if (indicesTensor.dim() == 0 && indicesTensor.numel() == 1) {
        gradTensor.unsqueeze(0);
    }
    if (gradTensor.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), gradTensor.dtype());
    }

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indicesDesc(indicesTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;

    DIOPI_CALLCNNL(cnnlGetEmbeddingBackwardWorkspaceSize(handle, gradDesc.get(), outDesc.get(), scaleGradByfreq, &workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlEmbeddingBackward(handle,
                                         paddingIdxCasted,
                                         scaleGradByfreq,
                                         indicesDesc.get(),
                                         indicesTensor.data(),
                                         gradDesc.get(),
                                         gradTensor.data(),
                                         workspace,
                                         workspaceSize,
                                         outDesc.get(),
                                         outTensorTmp.data()));
    if (outTensorTmp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl
