#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiSum(diopiContextHandle_t ctx,
                      diopiTensorHandle_t out,
                      diopiConstTensorHandle_t input,
                      diopiSize_t dim,
                      diopiDtype_t dtype) {
    /* Get handle and generate tensors */
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = impl::camb::DiopiTensor(input);
    auto output_tr = impl::camb::DiopiTensor(out);

    /* Some basic check */
    DIOPI_CHECK(input_tr.dtype() != diopi_dtype_int16 && input_tr.dtype() != diopi_dtype_int64 &&
        input_tr.dtype() != diopi_dtype_int8 && input_tr.dtype() != diopi_dtype_uint8, "Unsupported input dtype");
    DIOPI_CHECK(input_tr.dtype() == output_tr.dtype(), "input->dtype should equal to output->dtype");

    /* generate tensor desc */
    CnnlTensorDesc input_desc(input_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc;
    if (dim.len > 0 && dim.len != input_tr.shape().size()) {
        DIOPI_CALL(output_desc.set(output_tr, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> out_dims = {1};
        DIOPI_CALL(output_desc.set(output_tr, CNNL_LAYOUT_ARRAY, out_dims));
    }

    /* generate reduce desc */
    cnnlDataType_t cnnl_dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&cnnl_dtype, dtype));
    CnnlReduceDescriptor reduce_desc(input_tr, dim, CNNL_REDUCE_ADD,
    cnnl_dtype, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

    /* require workspace */
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, input_desc.get(), output_desc.get(), reduce_desc.get(), &workspace_size));
    void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

    DIOPI_CALLCNNL(
        cnnlReduce(handle, reduce_desc.get(), workspace_ptr,
        workspace_size, nullptr, input_desc.get(), input_tr.data(),
        0, nullptr, nullptr, output_desc.get(), output_tr.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
