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
    DiopiTensor out_tensor(*out);
    DiopiTensor counts_tensor(*counts);

    std::vector<DiopiTensor *> pTensors{&input_tensor, &out_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DIOPI_CALL(dataTypeCast(ctx, index_tensor, diopi_dtype_int32));

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc countsDesc(counts_tensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlUniqueDescriptor_t, cnnlCreateUniqueDescriptor, cnnlDestroyUniqueDescriptor> uniqueDesc;
    cnnlUniqueSort_t mode = sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_FORWARD;
    bool return_indices = true;

    if (mode == CNNL_UNSORT_FORWARD) {
        DIOPI_CHECK(*dim == -1,
                    "the dimension of input must be one-dimensional "
                    "when mode is CNNL_UNSORT_FORWARD");
    }

    DIOPI_CALLCNNL(cnnlSetUniqueDescriptor(uniqueDesc.get(), mode, *dim, return_indices, return_counts));
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetUniqueWorkSpace(handle, uniqueDesc.get(), inputDesc.get(), &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    std::vector<int64_t> out_num(1);
    DiopiTensor out_num_tensor = requiresTensor(ctx, out_num, diopi_dtype_int32);

    DIOPI_CALLCNNL(cnnlUnique_v2(handle,
                                 uniqueDesc.get(),
                                 inputDesc.get(),
                                 input_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 static_cast<int *>(out_num_tensor.data()),
                                 outDesc.get(),
                                 out_tensor.data(),
                                 indexDesc.get(),
                                 index_tensor.data(),
                                 countsDesc.get(),
                                 counts_tensor.data()));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl