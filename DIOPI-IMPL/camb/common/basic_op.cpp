#include "common.hpp"

namespace impl {
namespace camb {

template <typename T1, typename T2, typename T3>
diopiError_t cnnl_op_tensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t op_type, T1 alpha1, T2 alpha2,
                            T3 beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_casted = input;
    DiopiTensor other_casted = other;
    DiopiTensor output_casted = out;

    std::vector<DiopiTensor*> tensors{&input_casted, &other_casted, &output_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));

    cnnlDataType_t comp_type;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&comp_type, input_casted.dtype()));

    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> op_desc;

    DIOPI_CALLCNNL(cnnlSetOpTensorDescriptor(op_desc.get(), CNNL_OP_TENSOR_SUB, comp_type, CNNL_NOT_PROPAGATE_NAN));

    std::shared_ptr<void> alpha1_value = nullptr;
    std::shared_ptr<void> alpha2_value = nullptr;
    std::shared_ptr<void> beta_value = nullptr;

    if (DiopiDataType::isInteger(input_casted.dtype())) {
        alpha1_value = std::make_shared<int32_t>(alpha1);
        alpha2_value = std::make_shared<int32_t>(alpha2);
        beta_value = std::make_shared<int32_t>(beta);
    } else if (DiopiDataType::isFloatPoint(input_casted.dtype())) {
        alpha1_value = std::make_shared<float>(alpha1);
        alpha2_value = std::make_shared<float>(alpha2);
        beta_value = std::make_shared<float>(beta);
    } else {
        set_last_error_string("%s", "cnnl op tensor only support int or float type.\n");
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc input_desc(input_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_casted, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetOpTensorWorkspaceSize(handle, input_desc.get(), other_desc.get(), output_desc.get(), &workspace_size));

    void* workspace = nullptr;
    if (workspace_size != 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlOpTensor(handle,
                                op_desc.get(),
                                alpha1_value.get(),
                                input_desc.get(),
                                input_casted.data(),
                                alpha2_value.get(),
                                other_desc.get(),
                                other_casted.data(),
                                workspace,
                                workspace_size,
                                beta_value.get(),
                                output_desc.get(),
                                output_casted.data()));

    DIOPI_CALL(dataTypeCast(ctx, out, output_casted));
    return diopiSuccess;
}

// Explicitly instantiate the template function for use in other .cpp files.
template diopiError_t cnnl_op_tensor<double, double, double>(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out,
                                                             cnnlOpTensorDesc_t op_type, double alpha1, double alpha2, double beta);

}  // namespace camb
}  // namespace impl
