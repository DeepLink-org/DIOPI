/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t flattenTo2d(std::vector<int64_t> inDims, std::vector<int>& outDims) {
    outDims.resize(2);
    if (inDims.size() >= 2) {
        outDims[0] = std::accumulate(inDims.begin(), inDims.end() - 1, 1, std::multiplies<>());
        outDims[1] = inDims[inDims.size() - 1];
    } else {
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}

diopiError_t matmul(diopiContextHandle_t ctx, DiopiTensor inputA, DiopiTensor inputB, DiopiTensor inputBias, DiopiTensor output, bool transA, bool transB) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<int> inputShape, weightShape, outputShape;
    DIOPI_CALL(flattenTo2d(inputA.shape(), inputShape));
    DIOPI_CALL(flattenTo2d(inputB.shape(), weightShape));
    DIOPI_CALL(flattenTo2d(output.shape(), outputShape));

    CnnlTensorDesc aDesc, bDesc, biasDesc, outputDesc;
    DIOPI_CALL(aDesc.set(inputA, CNNL_LAYOUT_ARRAY, inputShape));
    DIOPI_CALL(bDesc.set(inputB, CNNL_LAYOUT_ARRAY, weightShape));
    DIOPI_CALL(outputDesc.set(output, CNNL_LAYOUT_ARRAY, outputShape));

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmulDesc;

    cnnlDataType_t compType;
    if (output.dtype() == diopi_dtype_float32) {
        compType = CNNL_DTYPE_FLOAT;
    } else if (output.dtype() == diopi_dtype_float16) {
        compType = CNNL_DTYPE_HALF;
    } else {
        setLastErrorString("%s", "matmul on support float or half.");
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE, &(compType), sizeof(cnnlDataType_t)));

    int32_t isTransa = 0;
    if (transA) {
        isTransa = 1;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_TRANSA, &(isTransa), sizeof(int32_t)));

    int32_t isTransb = 0;
    if (transB) {
        isTransb = 1;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_TRANSB, &(isTransb), sizeof(int32_t)));

    int32_t allowTf32I32 = 0;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_ALLOW_TF32, &(allowTf32I32), sizeof(int32_t)));

    int32_t useBeta = 0;
    float beta = 0.0;
    void* biasPtr = nullptr;
    if (inputBias.defined()) {
        useBeta = 1;
        beta = 1.0;
        biasPtr = inputBias.data();
        DIOPI_CALL(biasDesc.set(inputBias, CNNL_LAYOUT_ARRAY));
        DIOPI_CALLCNNL(cnnlExpand(handle, biasDesc.get(), inputBias.data(), outputDesc.get(), output.data()));
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_USE_BETA, &(useBeta), sizeof(int32_t)));

    size_t workspaceSize = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> heuristicResult;
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> algo;
    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                              matmulDesc.get(),
                                              aDesc.get(),
                                              bDesc.get(),
                                              outputDesc.get(),
                                              outputDesc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristicResult.get(),
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult.get(), algo.get(), &workspaceSize));

    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    float alphaDefault = 1.0;

    DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                 matmulDesc.get(),
                                 algo.get(),
                                 &alphaDefault,
                                 aDesc.get(),
                                 inputA.data(),
                                 bDesc.get(),
                                 inputB.data(),
                                 &beta,
                                 outputDesc.get(),
                                 output.data(),
                                 workspace,
                                 workspaceSize,
                                 outputDesc.get(),
                                 output.data()));

    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor biasTensor(bias);
    DiopiTensor outputTensor(out);
    DiopiTensor outTemp = outputTensor;

    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, outTemp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, weightTensor, diopi_dtype_float32));
        if (bias != nullptr) {
            DIOPI_CALL(dataTypeCast(ctx, biasTensor, diopi_dtype_float32));
        }
    }

    DIOPI_CALL(matmul(ctx, inputTensor, weightTensor, biasTensor, outTemp, false, true));
    if (outTemp.dtype() != outputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTemp));
    }
    return diopiSuccess;
}
extern "C" diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradInputTemp = gradInputTensor;
    DiopiTensor gradWeightTemp = gradWeightTensor;
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTemp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTemp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, gradOutputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, weightTensor, diopi_dtype_float32));
    }
    DiopiTensor biasTensor((diopiTensorHandle_t) nullptr);

    DIOPI_CALL(matmul(ctx, gradOutputTensor, inputTensor, biasTensor, gradWeightTemp, true, false));
    if (gradWeightTemp.dtype() != gradWeightTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightTemp));
    }
    DIOPI_CALL(matmul(ctx, gradOutputTensor, weightTensor, biasTensor, gradInputTemp, false, false));
    if (gradInputTemp.dtype() != gradInputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTemp));
    }

    if (gradBias != nullptr) {
        DiopiTensor biasGradTensor(gradBias);
        DiopiTensor biasGradTemp = biasGradTensor;
        if (biasGradTemp.dtype() == diopi_dtype_float64) {
            DIOPI_CALL(dataTypeCast(ctx, biasGradTemp, diopi_dtype_float32));
        }
        CnnlTensorDesc biasGradDesc;
        DIOPI_CALL(biasGradDesc.set(biasGradTemp, CNNL_LAYOUT_ARRAY));

        std::vector<int> outputShape;
        DIOPI_CALL(flattenTo2d(gradOutputTensor.shape(), outputShape));
        CnnlTensorDesc gradOutputDesc;
        DIOPI_CALL(gradOutputDesc.set(gradOutputTensor, CNNL_LAYOUT_ARRAY, outputShape));

        int channelAxis = 1;
        size_t workspaceSizeBias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, gradOutputDesc.get(), biasGradDesc.get(), channelAxis, &workspaceSizeBias))

        void* workspaceBias = nullptr;
        if (0 != workspaceSizeBias) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(
            handle, gradOutputDesc.get(), gradOutputTensor.data(), channelAxis, biasGradDesc.get(), biasGradTemp.data(), workspaceBias, workspaceSizeBias));
        if (biasGradTensor.dtype() != biasGradTemp.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, biasGradTensor, biasGradTemp));
        }
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
