#include "common.hpp"

namespace impl {
namespace camb {

template <typename T1, typename T2, typename T3>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t opType, T1 alpha1, T2 alpha2,
                          T3 beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<DiopiTensor*> tensors{&input, &other};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));

    DiopiTensor outputTmp = out;
    if (outputTmp.dtype() != input.dtype()) {
        outputTmp = requiresTensor(ctx, out.shape(), input.dtype());
    }

    cnnlDataType_t compType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&compType, input.dtype()));
    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> opDesc;
    DIOPI_CALL_CNNL(cnnlSetOpTensorDescriptor(opDesc.get(), opType, compType, CNNL_NOT_PROPAGATE_NAN));

    std::shared_ptr<void> alpha1Value = nullptr;
    std::shared_ptr<void> alpha2Value = nullptr;
    std::shared_ptr<void> betaValue = nullptr;

    if (DiopiDataType::isInteger(input.dtype())) {
        alpha1Value = std::make_shared<int32_t>(alpha1);
        alpha2Value = std::make_shared<int32_t>(alpha2);
        betaValue = std::make_shared<int32_t>(beta);
    } else if (DiopiDataType::isFloatPoint(input.dtype())) {
        alpha1Value = std::make_shared<float>(alpha1);
        alpha2Value = std::make_shared<float>(alpha2);
        betaValue = std::make_shared<float>(beta);
    } else {
        setLastErrorString("%s", "cnnl op tensor only support int or float type.\n");
        return diopiDtypeNotSupported;
    }

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(other, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetOpTensorWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outputDesc.get(), &workspaceSize));

    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALL_CNNL(cnnlOpTensor(handle,
                                 opDesc.get(),
                                 alpha1Value.get(),
                                 inputDesc.get(),
                                 input.data(),
                                 alpha2Value.get(),
                                 otherDesc.get(),
                                 other.data(),
                                 workspace,
                                 workspaceSize,
                                 betaValue.get(),
                                 outputDesc.get(),
                                 outputTmp.data()));

    DIOPI_CALL(dataTypeCast(ctx, out, outputTmp));
    return diopiSuccess;
}

// Explicitly instantiate the template function for use in other .cpp files.
template diopiError_t cnnlOpTensor<double, double, double>(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out,
                                                           cnnlOpTensorDesc_t op_type, double alpha1, double alpha2, double beta);

template <typename T = double>
diopiError_t cnnlTransformAdaptor(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, T other, T alpha) {
    auto handle = cnnlHandlePool.get(ctx);

    // std::vector<DiopiTensor *> inTensors{&input};
    // std::set<diopiDtype_t> supDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};
    // DIOPI_CALL(autoCastTensorType(ctx, inTensors, supDtypes));

    DiopiTensor outTmp = out;
    if (outTmp.dtype() != input.dtype()) {
        outTmp = requiresTensor(ctx, out.shape(), input.dtype());
    }

    std::shared_ptr<void> alp = nullptr;
    std::shared_ptr<void> bet = nullptr;
    if (DiopiDataType::isInteger(input.dtype())) {
        alp = std::make_shared<int32_t>(1);
        bet = std::make_shared<int32_t>(other * alpha);
    } else {
        alp = std::make_shared<float>(1);
        bet = std::make_shared<float>(other * alpha);
    }

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTmpDesc(outTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlTransform_v2(handle, CNNL_POINTER_MODE_HOST, alp.get(), inputDesc.get(), input.data(), bet.get(), outTmpDesc.get(), outTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out, outTmp));

    return diopiSuccess;
}

template diopiError_t cnnlTransformAdaptor<double>(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, double alpha, double beta);

// template
// diopiError_t cnnlTransformAdaptor<void>(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, void* alpha, void* beta);

}  // namespace camb
}  // namespace impl
