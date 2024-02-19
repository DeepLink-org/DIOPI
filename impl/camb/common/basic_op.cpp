#include "common.hpp"

namespace impl {
namespace camb {

enum class OpBroadcastType { noBroadcast, inputBroadcast, otherBroadcast, bothBroadcast };

OpBroadcastType checkOpBroadcast(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& otherShape) {
    int dimA = inputShape.size();
    int dimB = otherShape.size();
    int minDim;
    bool broadCastA = false;
    bool broadCastB = false;

    if (dimA == 0) {
        broadCastA = true;
    }

    if (dimB == 0) {
        broadCastB = true;
    }

    if (dimA > dimB) {
        minDim = dimB;
        broadCastB = true;
    } else if (dimA < dimB) {
        minDim = dimA;
        broadCastA = true;
    } else {
        minDim = dimA;
    }

    for (int i = 1; i <= minDim; i++) {
        if (inputShape[dimA - i] == otherShape[dimB - i]) {
            continue;
        } else if (inputShape[dimA - i] == 1) {
            broadCastA = true;
            continue;
        } else if (otherShape[dimB - i] == 1) {
            broadCastB = true;
            continue;
        } else {
            return OpBroadcastType::noBroadcast;
        }
    }
    if (broadCastA && broadCastB) {
        return OpBroadcastType::bothBroadcast;
    } else if (broadCastA) {
        return OpBroadcastType::inputBroadcast;
    } else if (broadCastB) {
        return OpBroadcastType::otherBroadcast;
    } else {
        return OpBroadcastType::noBroadcast;
    }
}

diopiError_t opBroadcastCast(const DiopiTensor& inputTensor, DiopiTensor& otherTensor, std::vector<int64_t>& targetShape, std::vector<int64_t>& targetStride,
                             bool& toPermuteFlag) {
    // get order of Tensor A
    // change shape and stride of Tensor B
    //  shape2,3,4,5 stride60,1,15,3 order0,3,1,2 reverseOrder0,2,3,1
    //  shape3,4,1 stride4,1,1 ->shape1,3,4,1 stride12,4,1,1 ->shape1,3,4,1,stride12,1,3,3 flag = true
    //  shape32,3,224,224 contiguous order0,1,2,3 reverseOrder0,1,2,3
    // shape3,1,1 contiguous ->shape1,3,1,1 stride3,3,1,1 ->shape1,3,1,1 stride3,3,1,1 ->flag = flase
    std::vector<int32_t> order(inputTensor.dim(), 0);
    std::vector<int32_t> reverseOrder(inputTensor.dim(), 0);
    getPermuteOrder(inputTensor, order, reverseOrder);
    targetShape = otherTensor.shape();
    std::vector<int64_t> curStride = otherTensor.stride();
    targetStride = inputTensor.stride();
    int firstStride = 1;
    if (otherTensor.dim() > 0) {
        firstStride = otherTensor.stride()[0] * otherTensor.stride()[0];
    }
    if (inputTensor.dim() > otherTensor.dim()) {
        targetShape.insert(targetShape.begin(), inputTensor.dim() - otherTensor.dim(), 1);
        curStride.insert(curStride.begin(), inputTensor.dim() - otherTensor.dim(), firstStride);
        otherTensor.asStrided(targetShape, curStride);
    }

    int cur = 1;
    for (int i = inputTensor.dim() - 1; i >= 0; i--) {
        targetStride[reverseOrder[i]] = cur;
        cur *= targetShape[reverseOrder[i]];
    }

    toPermuteFlag = true;
    if (curStride == targetStride) {
        toPermuteFlag = false;
    }

    return diopiSuccess;
}

template <typename T1, typename T2, typename T3>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor& input, DiopiTensor& other, DiopiTensor& out, cnnlOpTensorDesc_t opType, T1 alpha1, T2 alpha2,
                          T3 beta) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<DiopiTensor*> tensors{&input, &other};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32}));
    std::vector<int64_t> outTmpStride;
    std::vector<int64_t> outTmpShape;

    CnnlTensorDesc inputDesc;
    CnnlTensorDesc otherDesc;
    CnnlTensorDesc outputDesc;
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

        inputDesc.set(input, CNNL_LAYOUT_ARRAY);
        otherDesc.set(other, CNNL_LAYOUT_ARRAY);
        outputDesc.set(input.dtype(), outTmpShape, outTmpStride, CNNL_LAYOUT_ARRAY);

    } else {
        // in these cases, inputA & inputB should be broadcast operation
        OpBroadcastType broadcastType = checkOpBroadcast(input.shape(), other.shape());
        DIOPI_CHECK(broadcastType != OpBroadcastType::noBroadcast, "cannot broadcast input & other tensors");
        std::vector<int64_t> targetShape;
        std::vector<int64_t> targetStride;
        bool toPermuteFlag;
        if (input.isContiguous() && other.isContiguous()) {
            outTmpStride = out.stride();
            outTmpShape = out.shape();
            inputDesc.set(input, CNNL_LAYOUT_ARRAY);
            otherDesc.set(other, CNNL_LAYOUT_ARRAY);
            outputDesc.set(input.dtype(), outTmpShape, outTmpStride, CNNL_LAYOUT_ARRAY);

        } else if (broadcastType == OpBroadcastType::otherBroadcast) {
            opBroadcastCast(input, other, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor otherTmp = requiresTensor(ctx, targetShape, targetStride, other.dtype());
                DIOPI_CALL(permuteCopy(ctx, other, otherTmp));
                other = otherTmp;
            }
            std::vector<int32_t> order(input.dim(), 0);
            std::vector<int32_t> reverseOrder(input.dim(), 0);
            getPermuteOrder(input, order, reverseOrder);
            std::vector<int64_t> cnnlInShape = changeVecAccordingToOrder(input.shape(), reverseOrder);
            std::vector<int64_t> cnnlInStride = changeVecAccordingToOrder(input.stride(), reverseOrder);
            std::vector<int64_t> cnnlOtherShape = changeVecAccordingToOrder(targetShape, reverseOrder);
            std::vector<int64_t> cnnlOtherStride = changeVecAccordingToOrder(targetStride, reverseOrder);
            inputDesc.set(input.dtype(), cnnlInShape, cnnlInStride, CNNL_LAYOUT_ARRAY);
            otherDesc.set(input.dtype(), cnnlOtherShape, cnnlOtherStride, CNNL_LAYOUT_ARRAY);
            outputDesc.set(input.dtype(), cnnlInShape, cnnlInStride, CNNL_LAYOUT_ARRAY);

            outTmpStride = input.stride();
            outTmpShape = input.shape();
        } else if (broadcastType == OpBroadcastType::inputBroadcast) {
            opBroadcastCast(other, input, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor inputTmp = requiresTensor(ctx, targetShape, targetStride, input.dtype());
                DIOPI_CALL(permuteCopy(ctx, input, inputTmp));
                input = inputTmp;
            }
            std::vector<int32_t> order(other.dim(), 0);
            std::vector<int32_t> reverseOrder(other.dim(), 0);
            getPermuteOrder(other, order, reverseOrder);
            std::vector<int64_t> cnnlInShape = changeVecAccordingToOrder(targetShape, reverseOrder);
            std::vector<int64_t> cnnlInStride = changeVecAccordingToOrder(targetStride, reverseOrder);
            std::vector<int64_t> cnnlOtherShape = changeVecAccordingToOrder(other.shape(), reverseOrder);
            std::vector<int64_t> cnnlOtherStride = changeVecAccordingToOrder(other.stride(), reverseOrder);
            inputDesc.set(input.dtype(), cnnlInShape, cnnlInStride, CNNL_LAYOUT_ARRAY);
            otherDesc.set(input.dtype(), cnnlOtherShape, cnnlOtherStride, CNNL_LAYOUT_ARRAY);
            outputDesc.set(input.dtype(), cnnlOtherShape, cnnlOtherStride, CNNL_LAYOUT_ARRAY);

            outTmpStride = other.stride();
            outTmpShape = other.shape();
        } else {
            // it can be improved in the future, how strides and shapes can best accelerate "cnnlOpTensor"
            // e.g. shape(3,1)+shape(1,5)
            DIOPI_CALL(contiguous(ctx, input));
            DIOPI_CALL(contiguous(ctx, other));
            outTmpShape = out.shape();
            outTmpStride = calContiguousStride(outTmpShape);
            inputDesc.set(input, CNNL_LAYOUT_ARRAY);
            otherDesc.set(other, CNNL_LAYOUT_ARRAY);
            outputDesc.set(input.dtype(), outTmpShape, outTmpStride, CNNL_LAYOUT_ARRAY);
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
    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
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
        OpBroadcastType broadcastType = checkOpBroadcast(inputTensor.shape(), otherTensor.shape());
        DIOPI_CHECK(broadcastType != OpBroadcastType::noBroadcast, "cannot broadcast input & other tensors");
        std::vector<int64_t> targetShape;
        std::vector<int64_t> targetStride;
        bool toPermuteFlag;
        if (inputTensor.isContiguous() && otherTensor.isContiguous()) {
            outTmpShape = outTensor.shape();
            outTmpStride = outTensor.stride();
        } else if (broadcastType == OpBroadcastType::otherBroadcast) {
            opBroadcastCast(inputTensor, otherTensor, targetShape, targetStride, toPermuteFlag);
            if (toPermuteFlag) {
                DiopiTensor otherTmp = requiresTensor(ctx, targetShape, targetStride, otherTensor.dtype());
                DIOPI_CALL(permuteCopy(ctx, otherTensor, otherTmp));
                otherTensor = otherTmp;
            }
            outTmpStride = inputTensor.stride();
            outTmpShape = inputTensor.shape();
        } else if (broadcastType == OpBroadcastType::inputBroadcast) {
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
    if (outTensorTemp.dtype() != inputTensor.dtype() || outTmpStride != outTensor.stride()) {
        outTensorTemp = requiresTensor(ctx, outTmpShape, outTmpStride, inputTensor.dtype());
    }

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
