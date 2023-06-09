#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, const int64_t *dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t *counts) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor index_tensor(indices);

    std::vector<DiopiTensor *> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DIOPI_CALL(dataTypeCast(ctx, index_tensor, diopi_dtype_int32));

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlUniqueDescriptor_t, cnnlCreateUniqueDescriptor, cnnlDestroyUniqueDescriptor> uniqueDesc;
    cnnlUniqueDescriptor_t unique_desc = uniqueDesc.get();
    cnnlUniqueSort_t mode = sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_FORWARD;
    bool return_indices = true;

    if (mode == CNNL_UNSORT_FORWARD) {
        DIOPI_CHECK(*dim == -1,
                    "the dimension of input must be one-dimensional "
                    "when mode is CNNL_UNSORT_FORWARD");
    }

    DIOPI_CALLCNNL(cnnlSetUniqueDescriptor(unique_desc, mode, *dim, return_indices, return_counts));
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetUniqueWorkspaceSize(handle, unique_desc, inputDesc.get(), &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    std::vector<int64_t> temp(1);
    DiopiTensor outlen_tensor = requiresTensor(ctx, temp, diopi_dtype_int32);
    DIOPI_CALLCNNL(cnnlUniqueGetOutLen(handle, unique_desc, inputDesc.get(), input_tensor.data(), workspace, (int *)(outlen_tensor.data())));

    syncStreamInCtx(ctx);
    int32_t outlen_host = 0;
    cnrtMemcpy(&outlen_host, outlen_tensor.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);

    std::vector<int64_t> out_shape(outlen_host);
    //  requiresTensor for output Tensor
    DiopiTensor counts_tensor = requiresTensor(ctx, out_shape, input_tensor.dtype());
    CnnlTensorDesc countsDesc(counts_tensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor out_tensor = requiresTensor(ctx, out_shape, input_tensor.dtype());
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlUnique_v2(handle,
                                 unique_desc,
                                 inputDesc.get(),
                                 input_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 static_cast<int *>(outlen_tensor.data()),
                                 outDesc.get(),
                                 out_tensor.data(),
                                 indexDesc.get(),
                                 index_tensor.data(),
                                 countsDesc.get(),
                                 counts_tensor.data()));

    *out = diopiTensorHandle_t(out_tensor);
    if (return_counts) {
        *counts = diopiTensorHandle_t(counts_tensor);
    }
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl