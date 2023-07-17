/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiLinalgQR(diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char *mode, diopiTensorHandle_t Q, diopiTensorHandle_t R) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor output1Tensor;
    DiopiTensor output2Tensor(R);
    DiopiTensor inputTensor(A);
    DIOPI_CHECK(inputTensor.dim() >= 2 && inputTensor.dim() <= 8,
                "The number of dimensions of each input tensor should be equal to or greater than 2, and no greater than 8.");
    std::vector<DiopiTensor *> tensor{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, supportedDtypes));

    bool reduced = (strcmp(mode, "r") == 0 || strcmp(mode, "reduced") == 0) ? true : false;
    if (strcmp(mode, "r") == 0) {
        std::vector<int64_t> output1Shape(inputTensor.shape().begin(), inputTensor.shape().end());
        int64_t m = inputTensor.shape()[inputTensor.dim() - 2];
        int64_t n = inputTensor.shape()[inputTensor.dim() - 1];
        output1Shape[output1Shape.size() - 1] = std::min(m, n);
        output1Tensor = requiresTensor(ctx, output1Shape, inputTensor.dtype());
    } else {
        output1Tensor = DiopiTensor(Q);
    }

    DiopiTensor output1TensorTmp = output1Tensor;
    DiopiTensor output2TensorTmp = output2Tensor;
    if (inputTensor.dtype() != output1Tensor.dtype() || inputTensor.dtype() != output2Tensor.dtype()) {
        output1TensorTmp = requiresTensor(ctx, output1Tensor.shape(), inputTensor.dtype());
        output2TensorTmp = requiresTensor(ctx, output2Tensor.shape(), inputTensor.dtype());
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output1Desc(output1TensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output2Desc(output2TensorTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetQRWorkspaceSize(handle, inputDesc.get(), reduced, &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlQR(handle,
                          inputDesc.get(),
                          inputTensor.data(),
                          output1Desc.get(),
                          output1TensorTmp.data(),
                          output2Desc.get(),
                          output2TensorTmp.data(),
                          workspace,
                          workspaceSize,
                          reduced));
    if (output1TensorTmp.dtype() != output1Tensor.dtype() || output2TensorTmp.dtype() != output2Tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output1Tensor, output1TensorTmp));
        DIOPI_CALL(dataTypeCast(ctx, output2Tensor, output2TensorTmp));
    }
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
