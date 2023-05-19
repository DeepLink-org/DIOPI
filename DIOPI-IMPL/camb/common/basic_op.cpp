#include "common.hpp"

namespace impl {
namespace camb {

template <typename T1, typename T2, typename T3>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t opType, T1 alpha1, T2 alpha2,
                            T3 beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputCasted = input;
    DiopiTensor otherCasted = other;
    DiopiTensor outputCasted = out;

    std::vector<DiopiTensor*> tensors{&inputCasted, &otherCasted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));

    cnnlDataType_t compType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&compType, inputCasted.dtype()));

    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> opDesc;

    DIOPI_CALLCNNL(cnnlSetOpTensorDescriptor(opDesc.get(), CNNL_OP_TENSOR_SUB, compType, CNNL_NOT_PROPAGATE_NAN));

    std::shared_ptr<void> alpha1Value = nullptr;
    std::shared_ptr<void> alpha2Value = nullptr;
    std::shared_ptr<void> betaValue = nullptr;

    if (DiopiDataType::isInteger(inputCasted.dtype())) {
        alpha1Value = std::make_shared<int32_t>(alpha1);
        alpha2Value = std::make_shared<int32_t>(alpha2);
        betaValue = std::make_shared<int32_t>(beta);
    } else if (DiopiDataType::isFloatPoint(inputCasted.dtype())) {
        alpha1Value = std::make_shared<float>(alpha1);
        alpha2Value = std::make_shared<float>(alpha2);
        betaValue = std::make_shared<float>(beta);
    } else {
        set_last_error_string("%s", "cnnl op tensor only support int or float type.\n");
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc inputDesc(inputCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputCasted, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetOpTensorWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outputDesc.get(), &workspaceSize));

    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlOpTensor(handle,
                                opDesc.get(),
                                alpha1Value.get(),
                                inputDesc.get(),
                                inputCasted.data(),
                                alpha2Value.get(),
                                otherDesc.get(),
                                otherCasted.data(),
                                workspace,
                                workspaceSize,
                                betaValue.get(),
                                outputDesc.get(),
                                outputCasted.data()));

    DIOPI_CALL(dataTypeCast(ctx, out, outputCasted));
    return diopiSuccess;
}

// Explicitly instantiate the template function for use in other .cpp files.
template diopiError_t cnnlOpTensor<double, double, double>(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out,
                                                             cnnlOpTensorDesc_t op_type, double alpha1, double alpha2, double beta);

}  // namespace camb
}  // namespace impl
