#include "common.hpp"
namespace impl {
namespace camb {

template <typename T1, typename T2, typename T3>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor& input, DiopiTensor& other, DiopiTensor& out, cnnlOpTensorDesc_t opType, T1 alpha1, T2 alpha2,
                          T3 beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<DiopiTensor*> tensors{&input, &other};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));
    std::vector<int64_t> outTmpStride;
    std::vector<int64_t> outTmpShape;
    if (input.shape() == other.shape()) {
        // in these cases, inputA & inputB & output will have the same shape
        DIOPI_CHECK(input.shape() == out.shape(), "input shape should match output shape")
        if (input.stride() != other.stride()) {
            DiopiTensor otherTmp = requiresTensor(ctx, input.shape(), input.stride(), input.dtype());
            DIOPI_CALL(permuteCopy(ctx, other, otherTmp));
            other = otherTmp;
        }
        outTmpStride = input.stride();
        outTmpShape = input.shape();
    } else {
        // in these cases, inputA & inputB should be broadcast operation
        int broadcastType = isBroadcast(input, other);
        DIOPI_CHECK(broadcastType > 0, "cannot broadcast input & other tensors");
        std::vector<int64_t> targetShape;
        std::vector<int64_t> targetStride;
        bool toPermuteFlag;
        if (input.isContiguous() && other.isContiguous()) {
            outTmpStride = out.stride();
            outTmpShape = out.shape();
        } else if (broadcastType == 2) {
            opBroadcastCast(input, other, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor otherTmp = requiresTensor(ctx, targetShape, targetStride, other.dtype());
                DIOPI_CALL(permuteCopy(ctx, other, otherTmp));
                other = otherTmp;
            }
            outTmpStride = input.stride();
            outTmpShape = input.shape();
        } else if (broadcastType == 1) {
            opBroadcastCast(other, input, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor inputTmp = requiresTensor(ctx, targetShape, targetStride, input.dtype());
                DIOPI_CALL(permuteCopy(ctx, input, inputTmp));
                input = inputTmp;
            }
            outTmpStride = other.stride();
            outTmpShape = other.shape();
        } else {
            // it can be improved in the future, how strides and shapes can best accelerate "cnnlOpTensor"
            // e.g. shape(3,1)+shape(1,5)
            DIOPI_CALL(contiguous(ctx, input));
            DIOPI_CALL(contiguous(ctx, other));
            outTmpShape = out.shape();
            outTmpStride = calContiguousStride(outTmpShape);
        }
    }

    DiopiTensor outputTmp = out;
    if (outputTmp.dtype() != input.dtype() || outTmpStride != out.stride()) {
        outputTmp = requiresTensor(ctx, outTmpShape, outTmpStride, input.dtype());
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

    if (outputTmp.stride() != out.stride()) {
        DIOPI_CALL(diopiCopyInp(ctx, outputTmp.tensorHandle(), out.tensorHandle()));
    } else if (outputTmp.dtype() != out.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out, outputTmp));
    }
    return diopiSuccess;
}

// Explicitly instantiate the template function for use in other .cpp files.
template diopiError_t cnnlOpTensor<double, double, double>(diopiContextHandle_t ctx, DiopiTensor& input, DiopiTensor& other, DiopiTensor& out,
                                                           cnnlOpTensorDesc_t op_type, double alpha1, double alpha2, double beta);

template <typename T>
diopiError_t cnnlTransformAdaptor(diopiContextHandle_t ctx, DiopiTensor& out, DiopiTensor& input, T other, T alpha, T beta) {
    auto handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTmp = out;
    if (outTmp.dtype() != input.dtype() || outTmp.stride() != input.stride()) {
        outTmp = requiresTensor(ctx, out.shape(), input.stride(), input.dtype());
    }

    std::shared_ptr<void> alp = nullptr;
    std::shared_ptr<void> bet = nullptr;
    if (DiopiDataType::isInteger(input.dtype())) {
        alp = std::make_shared<int32_t>(beta);
        bet = std::make_shared<int32_t>(other * alpha);
    } else {
        alp = std::make_shared<float>(beta);
        bet = std::make_shared<float>(other * alpha);
    }

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTmpDesc(outTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlTransform_v2(handle, CNNL_POINTER_MODE_HOST, alp.get(), inputDesc.get(), input.data(), bet.get(), outTmpDesc.get(), outTmp.data()));

    if (outTmp.stride() != out.stride()) {
        DIOPI_CALL(diopiCopyInp(ctx, outTmp.tensorHandle(), out.tensorHandle()));
    } else if (outTmp.dtype() != out.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out, outTmp));
    }
    return diopiSuccess;
}

template diopiError_t cnnlTransformAdaptor<double>(diopiContextHandle_t ctx, DiopiTensor& out, DiopiTensor& input, double other, double alpha, double beta);

diopiError_t diopiDivInternal(diopiContextHandle_t ctx, DiopiTensor& inputTensor, DiopiTensor& otherTensor, DiopiTensor& outTensor,
                              diopiRoundMode_t roundingMode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<int64_t> outTmpStride;
    std::vector<int64_t> outTmpShape;
    if (inputTensor.shape() == otherTensor.shape()) {
        // in these cases, inputA & inputB & output will have the same shape
        DIOPI_CHECK(inputTensor.shape() == outTensor.shape(), "input shape should match output shape")
        if (inputTensor.stride() != otherTensor.stride()) {
            DiopiTensor otherTmp = requiresTensor(ctx, inputTensor.shape(), inputTensor.stride(), inputTensor.dtype());
            DIOPI_CALL(permuteCopy(ctx, otherTensor, otherTmp));
            otherTensor = otherTmp;
        }
        outTmpShape = inputTensor.shape();
        outTmpStride = inputTensor.stride();
    } else {
        // in these cases, inputA & inputB should be broadcast operation
        int broadcastType = isBroadcast(inputTensor, otherTensor);
        DIOPI_CHECK(broadcastType > 0, "cannot broadcast input & other tensors");
        std::vector<int64_t> targetShape;
        std::vector<int64_t> targetStride;
        bool toPermuteFlag;
        if (inputTensor.isContiguous() && otherTensor.isContiguous()) {
            outTmpShape = outTensor.shape();
            outTmpStride = outTensor.stride();
        } else if (broadcastType == 2) {
            opBroadcastCast(inputTensor, otherTensor, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor otherTmp = requiresTensor(ctx, targetShape, targetStride, otherTensor.dtype());
                DIOPI_CALL(permuteCopy(ctx, otherTensor, otherTmp));
                otherTensor = otherTmp;
            }
            outTmpStride = inputTensor.stride();
            outTmpShape = inputTensor.shape();
        } else if (broadcastType == 1) {
            opBroadcastCast(otherTensor, inputTensor, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor inputTmp = requiresTensor(ctx, targetShape, targetStride, inputTensor.dtype());
                DIOPI_CALL(permuteCopy(ctx, inputTensor, inputTmp));
                inputTensor = inputTmp;
            }
            outTmpStride = otherTensor.stride();
            outTmpShape = otherTensor.shape();

        } else {
            // it can be improved in the future, how strides and shapes can best accelerate "cnnlOpTensor"
            // e.g. shape(3,1)+shape(1,5)
            DIOPI_CALL(contiguous(ctx, inputTensor));
            DIOPI_CALL(contiguous(ctx, otherTensor));
            outTmpShape = outTensor.shape();
            outTmpStride = calContiguousStride(outTmpShape);
        }
    }

    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    cnnlComputationPreference_t preferFloor = CNNL_COMPUTATION_ULTRAHIGH_PRECISION;

    DiopiTensor outTensorTemp = outTensor;
    if (outTmpStride != outTensor.stride()) {
        outTensorTemp = requiresTensor(ctx, outTmpShape, outTmpStride, inputTensor.dtype());
    }
    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTensor, &outTensorTemp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);
    size_t workspaceSize = 0;
    void* workspace = nullptr;

    switch (roundingMode) {
        case RoundModeFloor:
            DIOPI_CALL_CNNL(cnnlGetFloorDivWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlFloorDiv_v2(handle,
                                            preferFloor,
                                            inputDesc.get(),
                                            inputTensor.data(),
                                            otherDesc.get(),
                                            otherTensor.data(),
                                            outDesc.get(),
                                            outTensorTemp.data(),
                                            workspace,
                                            workspaceSize));
            break;
        case RoundModeTrunc:
            DIOPI_CALL_CNNL(cnnlGetFloorDivTruncWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlFloorDivTrunc(handle,
                                              prefer,
                                              inputDesc.get(),
                                              inputTensor.data(),
                                              otherDesc.get(),
                                              otherTensor.data(),
                                              outDesc.get(),
                                              outTensorTemp.data(),
                                              workspace,
                                              workspaceSize));
            break;
        case RoundModeNone:
            DIOPI_CALL_CNNL(cnnlGetDivWorkspaceSize(handle, inputDesc.get(), otherDesc.get(), outDesc.get(), &workspaceSize));
            workspace = requiresBuffer(ctx, workspaceSize).data();
            DIOPI_CALL_CNNL(cnnlDiv_v2(handle,
                                       prefer,
                                       inputDesc.get(),
                                       inputTensor.data(),
                                       otherDesc.get(),
                                       otherTensor.data(),
                                       workspace,
                                       workspaceSize,
                                       outDesc.get(),
                                       outTensorTemp.data()));

            break;
        default:
            break;
    }
    if (outTensorTemp.stride() != outTensor.stride()) {
        DIOPI_CALL(diopiCopyInp(ctx, outTensorTemp.tensorHandle(), outTensor.tensorHandle()));
    } else if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
