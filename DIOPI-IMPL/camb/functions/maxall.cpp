/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(max);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECK(input_tensor.dtype() == output_tensor.dtype(), "input->dtype should equal to output->dtype");

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, input_tensor.dtype()));
    std::vector<int64_t> dims(input_tensor.dim());
    for (int i = 0; i < input_tensor.dim(); i++) {
        dims[i] = i;
    }
    diopiSize_t dim = {dims.data(), input_tensor.dim()};
    CnnlReduceDescriptor reduce_desc(input_tensor, dim, CNNL_REDUCE_MAX, dtype, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, input_desc.get(), output_desc.get(), reduce_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    size_t indices_size_inbytes(0);
    void* indices = nullptr;
    void* alpha = nullptr;
    void* beta = nullptr;
    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduce_desc.get(),
                              workspace,
                              workspace_size,
                              alpha,
                              input_desc.get(),
                              input_tensor.data(),
                              indices_size_inbytes,
                              indices,
                              beta,
                              output_desc.get(),
                              output_tensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
