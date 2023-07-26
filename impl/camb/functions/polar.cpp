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
DIOPI_API diopiError_t diopiPolar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor absTensor(abs);
    DiopiTensor angleTensor(angle);

    std::vector<DiopiTensor*> pTensors{&absTensor, &angleTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc absDesc(absTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc angleDesc(angleTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPolarWorkspaceSize(handle, absDesc.get(), angleDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(
        cnnlPolar(handle, absDesc.get(), absTensor.data(), angleDesc.get(), angleTensor.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));

    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
