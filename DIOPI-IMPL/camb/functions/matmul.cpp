#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
static std::vector<int> getPerm(DiopiTensor tensor, int64_t dim0, int64_t dim1) {
    int inputSize = tensor.shape().size();
    if (dim0 < 0) {
        dim0 = dim0 + inputSize;
    }
    if (dim1 < 0) {
        dim1 = dim1 + inputSize;
    }

    std::vector<int> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    return perms;
}

static std::vector<int64_t> inferSize(std::vector<int64_t> batchTensor1, std::vector<int64_t> batchTensor2) {
    if (batchTensor1.size() < batchTensor2.size()) {
        batchTensor1.insert(batchTensor1.begin(), batchTensor2.size() - batchTensor1.size(), 1);
    } else if (batchTensor1.size() > batchTensor2.size()) {
        batchTensor2.insert(batchTensor2.begin(), batchTensor1.size() - batchTensor2.size(), 1);
    }

    std::vector<int64_t> res(batchTensor1);
    for (int i = 0; i < batchTensor1.size(); i++) {
        if (1 == batchTensor1[i]) {
            res[i] = batchTensor2[i];
        }
    }

    return res;
}

static int64_t multiplyIntegers(std::vector<int64_t> tensor) {
    int64_t out = 1;
    for (long i : tensor) {
        out = out * i;
    }

    return out;
}

static diopiError_t vectorMulVector(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor vector1Tensor, DiopiTensor vector2Tensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (vector1Tensor.dtype() != diopi_dtype_float32 && vector1Tensor.dtype() != diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, vector1Tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, vector2Tensor, diopi_dtype_float32));
    }

    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vector1Desc(vector1Tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vector2Desc(vector2Tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor tempOut = requiresTensor(ctx, vector1Tensor.shape(), vector1Tensor.dtype());
    CnnlTensorDesc tempOutDesc(tempOut, CNNL_LAYOUT_ARRAY);

    std::vector<cnnlTensorDescriptor_t> inputsDesc(2);
    inputsDesc[0] = vector1Desc.get();
    inputsDesc[1] = vector2Desc.get();
    std::vector<const void*> inputs(2);
    inputs[0] = vector1Tensor.data();
    inputs[1] = vector2Tensor.data();

    DIOPI_CALLCNNL(cnnlMulN(handle, inputsDesc.data(), inputs.data(), 2, tempOutDesc.get(), tempOut.data()));
    int64_t dimData = 0;
    diopiSize_t dim = {&dimData, 1};

    if (outTensor.dtype() == vector1Tensor.dtype()) {
        DIOPI_CALL(diopiSum(ctx, (diopiTensorHandle_t)outTensor, (diopiTensorHandle_t)tempOut, dim));
    } else {
        DiopiTensor out32Tensor = requiresTensor(ctx, outTensor.shape(), vector1Tensor.dtype());
        DIOPI_CALL(diopiSum(ctx, (diopiTensorHandle_t)out32Tensor, (diopiTensorHandle_t)tempOut, dim));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
    }
    return diopiSuccess;
}

static diopiError_t matMulMat(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, DiopiTensor other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (input.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, other, diopi_dtype_float32));
    }

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(other, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmulDescGuard;
    cnnlMatMulDescriptor_t matmulDesc = matmulDescGuard.get();
    int32_t allowTf32I32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_ALLOW_TF32, &(allowTf32I32), sizeof(int32_t)));
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> matmulAlgo;
    cnnlMatMulAlgo_t algo = matmulAlgo.get();

    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> matMulHeuristic;
    cnnlMatMulHeuristicResult_t heuristicResult = matMulHeuristic.get();
    int returnAlgoCount = 0;
    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(
        handle, matmulDesc, inputDesc.get(), otherDesc.get(), outDesc.get(), outDesc.get(), nullptr, 1, &heuristicResult, &returnAlgoCount));
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult, algo, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    float alpha = 1;
    float beta = 0;
    if (out.dtype() == input.dtype()) {
        DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                     matmulDesc,
                                     algo,
                                     &alpha,
                                     inputDesc.get(),
                                     input.data(),
                                     otherDesc.get(),
                                     other.data(),
                                     &beta,
                                     outDesc.get(),
                                     out.data(),
                                     workspace,
                                     workspaceSize,
                                     outDesc.get(),
                                     out.data()));
    } else {
        DiopiTensor outTemp = requiresTensor(ctx, out.shape(), input.dtype());
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                     matmulDesc,
                                     algo,
                                     &alpha,
                                     inputDesc.get(),
                                     input.data(),
                                     otherDesc.get(),
                                     other.data(),
                                     &beta,
                                     outTempDesc.get(),
                                     outTemp.data(),
                                     workspace,
                                     workspaceSize,
                                     outTempDesc.get(),
                                     outTemp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out, outTemp));
    }

    return diopiSuccess;
}

static diopiError_t matMulVector(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, DiopiTensor vectorTensor) {
    if (inputTensor.shape()[1] != vectorTensor.shape()[0]) {
        vectorTensor.reshape({1, vectorTensor.shape()[0]});
        outTensor.reshape({vectorTensor.shape()[0], 1});
    } else {
        vectorTensor.reshape({vectorTensor.shape()[0], 1});
        outTensor.reshape({inputTensor.shape()[0], 1});
    }

    DIOPI_CALL(matMulMat(ctx, outTensor, inputTensor, vectorTensor));
    return diopiSuccess;
}

static diopiError_t transposeInternal(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    diopiTensorHandle_t out = (diopiTensorHandle_t)outTensor;

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> cnnlTransposeDesc;
    cnnlTransposeDescriptor_t transposeDesc = cnnlTransposeDesc.get();
    std::vector<int> perms = getPerm(input, dim0, dim1);
    cnnlSetTransposeDescriptor(transposeDesc, perms.size(), perms.data());

    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transposeDesc, &workspaceSize);
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    cnnlTranspose_v2(handle, transposeDesc, inputDesc.get(), input.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize);
    return diopiSuccess;
}

static diopiError_t batchMatmul(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, DiopiTensor otherTensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, otherTensor, diopi_dtype_float32));
    }

    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);

    int32_t allowTf32Int = 1;
    CnnlDescBase<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> bmmDescGuard;
    cnnlSetMatMulDescAttr(bmmDescGuard.get(), CNNL_MATMUL_ALLOW_TF32, &allowTf32Int, sizeof(allowTf32Int));
    CnnlDescBase<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> bmmAlgo;
    CnnlDescBase<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> bmmHeuristicResult;

    int returnAlgoCount = 0;
    cnnlGetBatchMatMulAlgoHeuristic(
        handle, bmmDescGuard.get(), inputDesc.get(), otherDesc.get(), outDesc.get(), nullptr, 1, &(bmmHeuristicResult.get()), &returnAlgoCount);

    size_t workspaceSize(0);
    cnnlGetBatchMatMulHeuristicResult(bmmHeuristicResult.get(), bmmAlgo.get(), &workspaceSize);
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                               bmmDescGuard.get(),
                                               bmmAlgo.get(),
                                               nullptr,
                                               inputDesc.get(),
                                               inputTensor.data(),
                                               otherDesc.get(),
                                               otherTensor.data(),
                                               nullptr,
                                               outDesc.get(),
                                               outTensor.data(),
                                               workspace,
                                               workspaceSize));
    } else {
        DiopiTensor outTemp = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                               bmmDescGuard.get(),
                                               bmmAlgo.get(),
                                               nullptr,
                                               inputDesc.get(),
                                               inputTensor.data(),
                                               otherDesc.get(),
                                               otherTensor.data(),
                                               nullptr,
                                               outTempDesc.get(),
                                               outTemp.data(),
                                               workspace,
                                               workspaceSize));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }

    return diopiSuccess;
}

static diopiError_t tensorMatmulTensor(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, DiopiTensor otherTensor) {
    if (inputTensor.dim() == 1 && otherTensor.dim() == 1) {
        DIOPI_CALL(vectorMulVector(ctx, outTensor, inputTensor, otherTensor));
        return diopiSuccess;
    } else if (inputTensor.dim() == 2 && otherTensor.dim() == 1) {
        DIOPI_CALL(matMulVector(ctx, outTensor, inputTensor, otherTensor));
        return diopiSuccess;
    } else if (inputTensor.dim() == 1 && otherTensor.dim() == 2) {
        std::vector<int64_t> shape(otherTensor.shape());
        shape[0] = otherTensor.shape()[1];
        shape[1] = otherTensor.shape()[0];
        DiopiTensor otherT = requiresTensor(ctx, shape, otherTensor.dtype());
        DIOPI_CALL(transposeInternal(ctx, otherT, otherTensor, 0, 1))
        DIOPI_CALL(matMulVector(ctx, outTensor, otherT, inputTensor));
        return diopiSuccess;
    } else if (inputTensor.dim() == 2 && otherTensor.dim() == 2) {
        DIOPI_CALL(matMulMat(ctx, outTensor, inputTensor, otherTensor));
        return diopiSuccess;
    } else if (inputTensor.dim() >= 3 && (otherTensor.dim() == 1 || otherTensor.dim() == 2)) {
        std::vector<int64_t> outputSize;
        outputSize.insert(outputSize.end(), inputTensor.shape().begin(), inputTensor.shape().end() - 1);
        if (otherTensor.dim() == 1) {
            std::vector<int64_t> tempShape(2);
            tempShape[0] = otherTensor.shape()[0];
            tempShape[1] = 1;
            otherTensor.reshape(tempShape);
        } else {
            outputSize.push_back(otherTensor.shape()[1]);
        }

        std::vector<int64_t> shape(2);
        shape[1] = inputTensor.shape()[inputTensor.dim() - 1];
        shape[0] = inputTensor.numel() / shape[1];
        inputTensor.reshape(shape);
        shape[1] = otherTensor.shape()[1];
        outTensor.reshape(shape);
        DIOPI_CALL(matMulMat(ctx, outTensor, inputTensor, otherTensor));
        return diopiSuccess;
    } else if ((inputTensor.dim() == 1 || inputTensor.dim() == 2) && otherTensor.dim() >= 3) {
        int inputDim = inputTensor.dim();
        int64_t n = inputTensor.dim() == 2 ? inputTensor.shape()[0] : 1;
        int64_t m = inputTensor.shape()[inputTensor.dim() - 1];
        int64_t p = otherTensor.shape()[otherTensor.dim() - 1];
        if (inputDim == 1) {
            inputTensor.reshape({n, m});
        }

        std::vector<int64_t> otherShape(otherTensor.shape());
        otherShape[otherTensor.shape().size() - 1] = otherTensor.shape()[otherTensor.shape().size() - 2];
        otherShape[otherTensor.shape().size() - 2] = otherTensor.shape()[otherTensor.shape().size() - 1];
        DiopiTensor otherTTensor = requiresTensor(ctx, otherShape, otherTensor.dtype());
        DIOPI_CALL(transposeInternal(ctx, otherTTensor, otherTensor, -1, -2))
        std::vector<int64_t> inputShape(inputTensor.shape());
        inputShape[0] = inputTensor.shape()[1];
        inputShape[1] = inputTensor.shape()[0];
        DiopiTensor inputTTensor = requiresTensor(ctx, inputShape, inputTensor.dtype());
        DIOPI_CALL(transposeInternal(ctx, inputTTensor, inputTensor, 0, 1))

        if (inputDim == 1) {
            DIOPI_CALL(tensorMatmulTensor(ctx, outTensor, otherTTensor, inputTTensor));
        } else {
            std::vector<int64_t> shape(otherTTensor.shape().begin(), otherTTensor.shape().end() - 1);
            shape.push_back(inputTensor.shape()[0]);
            DiopiTensor outTemp = requiresTensor(ctx, shape, outTensor.dtype());

            DIOPI_CALL(tensorMatmulTensor(ctx, outTemp, otherTTensor, inputTTensor));
            DIOPI_CALL(transposeInternal(ctx, outTensor, outTemp, -1, -2));
        }

        return diopiSuccess;
    } else if ((inputTensor.dim() >= 1 && otherTensor.dim() >= 1) && (inputTensor.dim() >= 3 || otherTensor.dim() >= 3)) {
        int64_t n = inputTensor.dim() > 1 ? inputTensor.shape()[inputTensor.dim() - 2] : 1;
        int64_t m1 = inputTensor.shape()[inputTensor.dim() - 1];
        int64_t dataLen = inputTensor.dim() > 2 ? inputTensor.shape().size() - 2 : 0;
        std::vector<int64_t> batchTensor1(inputTensor.shape().begin(), inputTensor.shape().begin() + dataLen);

        int64_t m2 = otherTensor.dim() > 1 ? otherTensor.shape()[inputTensor.dim() - 2] : 1;
        int64_t p = otherTensor.shape()[otherTensor.dim() - 1];
        dataLen = otherTensor.dim() > 2 ? otherTensor.shape().size() - 2 : 0;
        std::vector<int64_t> batchTensor2(otherTensor.shape().begin(), otherTensor.shape().begin() + dataLen);

        std::vector<int64_t> expandBatchPortion = inferSize(batchTensor1, batchTensor2);
        std::vector<int64_t> tensor1ExpandSize(expandBatchPortion);
        tensor1ExpandSize.insert(tensor1ExpandSize.end(), {n, m1});
        std::vector<int64_t> tensor2ExpandSize(expandBatchPortion);
        tensor2ExpandSize.insert(tensor2ExpandSize.end(), {m2, p});

        int64_t expandBatchProduct = multiplyIntegers(expandBatchPortion);
        std::vector<int64_t> tensor1BmmView({expandBatchProduct});
        tensor1BmmView.insert(tensor1BmmView.end(), {n, m1});
        std::vector<int64_t> tensor2BmmView({expandBatchProduct});
        tensor2BmmView.insert(tensor2BmmView.end(), {m2, p});

        DiopiTensor inputExpand = requiresTensor(ctx, tensor1ExpandSize, inputTensor.dtype());
        DiopiTensor otherExpand = requiresTensor(ctx, tensor2ExpandSize, otherTensor.dtype());
        broadcast(ctx, inputExpand, inputTensor);
        broadcast(ctx, otherExpand, otherTensor);
        inputExpand.reshape(tensor1BmmView);
        otherExpand.reshape(tensor2BmmView);

        std::vector<int64_t> outputShape({expandBatchProduct});
        if (inputTensor.dim() > 1) {
            outputShape.push_back(n);
        }
        if (otherTensor.dim() > 1) {
            outputShape.push_back(p);
        }
        outTensor.reshape(outputShape);
        DIOPI_CALL(batchMatmul(ctx, outTensor, inputExpand, otherExpand));
        return diopiSuccess;
    }

    setLastErrorString("both arguments to matmul need to be at least 1D, but they are ", inputTensor.dim(), "D and ", otherTensor.dim(), "D");
    return diopiErrorOccurred;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor outTensor(out);

    DIOPI_CALL(tensorMatmulTensor(ctx, outTensor, inputTensor, otherTensor));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
