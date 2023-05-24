#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    std::vector<int> shiftsTmp{shifts.data, shifts.data + shifts.len};
    std::vector<int> dimsTmp{dims.data, dims.data + dims.len};
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetRollWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlRoll(handle,
                            inputDesc.get(),
                            inputTensor.data(),
                            shiftsTmp.data(),
                            shiftsTmp.size(),
                            !dimsTmp.empty() ? dimsTmp.data() : nullptr,
                            dimsTmp.size(),
                            workspace,
                            workspaceSize,
                            outDesc.get(),
                            outTensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
