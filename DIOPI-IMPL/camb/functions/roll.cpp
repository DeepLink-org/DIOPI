#include <diopi/functions.h>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int> shifts_{shifts.data, shifts.data + shifts.len};
    std::vector<int> dims_{dims.data, dims.data + dims.len};
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetRollWorkspaceSize(handle, input_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlRoll(handle,
                            input_desc.get(),
                            input_tensor.data(),
                            shifts_.data(),
                            shifts_.size(),
                            dims_.size() > 0 ? dims_.data() : nullptr,
                            dims_.size(),
                            workspace,
                            workspace_size,
                            out_desc.get(),
                            out_tensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
