/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <vector>

#include "../aclnn/adaptor.hpp"
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

AscendTensor torchContextAttention(diopiContextHandle_t ctx, AscendTensor xq, AscendTensor xk, AscendTensor xv, int batchSize, int seqLen, int head, int dim) {
    xq.view({batchSize, seqLen, head, dim});
    AscendTensor xqTransposeAt;
    std::vector<int64_t> xqTransposeAtShape(xq.shape());
    int64_t tmp = xqTransposeAtShape[1];
    xqTransposeAtShape[1] = xqTransposeAtShape[2];
    xqTransposeAtShape[2] = tmp;
    makeTensor(ctx, xqTransposeAt, xqTransposeAtShape, xq.dtype());
    std::vector<int64_t> xqTransposeDims = {0, 2, 1, 3};
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, xq, xqTransposeDims, xqTransposeAt);

    xk.view({batchSize, seqLen, head, dim});
    AscendTensor xkTransposeAt;
    makeTensor(ctx, xkTransposeAt, xqTransposeAtShape, xk.dtype());
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, xk, xqTransposeDims, xkTransposeAt);

    xv.view({batchSize, seqLen, head, dim});
    AscendTensor xvTransposeAt;
    makeTensor(ctx, xvTransposeAt, xqTransposeAtShape, xv.dtype());
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, xv, xqTransposeDims, xvTransposeAt);

    AscendTensor maskAt;
    makeTensor(ctx, maskAt, {1, 1, seqLen, seqLen}, diopi_dtype_float32);
    AscendTensor onesMatrixAt;
    makeTensor(ctx, onesMatrixAt, {seqLen, seqLen}, diopi_dtype_float32);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, onesMatrixAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTril, ctx, onesMatrixAt, 0);
    maskAt = onesMatrixAt.unsqueeze(0).unsqueeze(0);
    diopiScalar_t valueScalar = constructDiopiScalarT(diopi_dtype_float32, -100000000.0);
    AscendTensor maskMatrixAt;
    makeTensor(ctx, maskMatrixAt, maskAt.shape(), diopi_dtype_int32);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, maskMatrixAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceTriu, ctx, maskMatrixAt, 1);
    AscendTensor maskBoolMatrixAt;
    makeTensor(ctx, maskBoolMatrixAt, maskMatrixAt.shape(), diopi_dtype_bool);
    DIOPI_ASCEND_CALL_ACLNN(aclnnCast, ctx, maskMatrixAt, diopi_dtype_bool, maskBoolMatrixAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceMaskedFillScalar, ctx, maskAt, maskBoolMatrixAt, &valueScalar);
    AscendTensor maskRepeatAt;
    std::vector<int64_t> maskRepeatAtShape(maskAt.shape());
    maskRepeatAtShape[0] *= batchSize;
    maskRepeatAtShape[1] *= head;
    makeTensor(ctx, maskRepeatAt, maskRepeatAtShape, maskAt.dtype());
    std::vector<int64_t> repeats = {batchSize, head, 1, 1};
    DIOPI_ASCEND_CALL_ACLNN(aclnnRepeat, ctx, maskAt, repeats, maskRepeatAt);

    AscendTensor scoresAt;
    AscendTensor xkTransposeAt2;
    std::vector<int64_t> xkTransposeAtShape(xkTransposeAt.shape());
    tmp = xkTransposeAtShape[2];
    xkTransposeAtShape[2] = xkTransposeAtShape[3];
    xkTransposeAtShape[3] = tmp;
    makeTensor(ctx, xkTransposeAt2, xkTransposeAtShape, xk.dtype());
    std::vector<int64_t> xkTransposeAt2Dims = {0, 1, 3, 2};
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, xkTransposeAt, xkTransposeAt2Dims, xkTransposeAt2);
    std::vector<int64_t> scoresShapeAt = xqTransposeAt.shape();
    scoresShapeAt[3] = xkTransposeAtShape[3];
    makeTensor(ctx, scoresAt, scoresShapeAt, xq.dtype());
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, xqTransposeAt, xkTransposeAt2, scoresAt, (int8_t)0);
    diopiScalar_t otherScalar = constructDiopiScalarT(diopi_dtype_float32, std::sqrt(dim));
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDivs, ctx, scoresAt, &otherScalar);

    AscendTensor adjustedScoresAt;
    std::vector<int64_t> adjustedScoresAtShape = inferSize(scoresAt.shape(), maskRepeatAt.shape());
    makeTensor(ctx, adjustedScoresAt, adjustedScoresAtShape, scoresAt.dtype());
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float32, 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, scoresAt, maskRepeatAt, &alphaScalar, adjustedScoresAt);
    DIOPI_ASCEND_CALL_ACLNN(aclnnSoftmax, ctx, adjustedScoresAt, adjustedScoresAt.dim() - 1, adjustedScoresAt);
    AscendTensor outputAt;
    std::vector<int64_t> outputAtShape = adjustedScoresAt.shape();
    outputAtShape[3] = xvTransposeAt.shape(3);
    makeTensor(ctx, outputAt, outputAtShape, scoresAt.dtype());
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, adjustedScoresAt, xvTransposeAt, outputAt, (int8_t)0);
    tmp = outputAtShape[1];
    outputAtShape[1] = outputAtShape[2];
    outputAtShape[2] = tmp;
    AscendTensor outputTransposeAt;
    makeTensor(ctx, outputTransposeAt, outputAtShape, outputAt.dtype());
    std::vector<int64_t> outputTransposeDims = {0, 2, 1, 3};
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, outputAt, outputTransposeDims, outputTransposeAt);
    outputTransposeAt.view({outputTransposeAt.numel() / static_cast<int64_t>(head * dim), head, dim});

    return outputTransposeAt;
}

diopiError_t diopiContextAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                            diopiConstTensorHandle_t v, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen, int maxInputLen) {
    AscendTensor bStartLocAt(bStartLoc);
    AscendTensor qAt(q);
    AscendTensor bSeqLenAt(bSeqLen);

    int batch = bStartLocAt.shape()[0];
    int head = qAt.shape()[1];
    int dim = qAt.shape()[2];

    void* bStartLocDataPtr = ascendTensorDeviceToHost(ctx, bStartLocAt);
    void* bSeqLenDataPtr = ascendTensorDeviceToHost(ctx, bSeqLenAt);
    for (int i = 0; i < batch; ++i) {
        int start = reinterpret_cast<int*>(bStartLocDataPtr)[i];
        int end = start + reinterpret_cast<int*>(bSeqLenDataPtr)[i];

        AscendTensor sliceAt;
        std::vector<int64_t> sliceAtShape(1, end - start);
        makeTensor(ctx, sliceAt, sliceAtShape, diopi_dtype_int32);
        diopiScalar_t startIndexScalar = constructDiopiScalarT(diopi_dtype_int32, start);
        diopiScalar_t endIndexScalar = constructDiopiScalarT(diopi_dtype_int32, end);
        diopiScalar_t stepScalar = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &startIndexScalar, &endIndexScalar, &stepScalar, sliceAt);

        diopiTensorHandle_t qIndex;
        diopiConstTensorHandle_t sliceTensorHandle = sliceAt.tensorHandle();
        ascend_npu::diopiIndex(ctx, &qIndex, q, &sliceTensorHandle, 1);

        diopiTensorHandle_t kIndex;
        ascend_npu::diopiIndex(ctx, &kIndex, k, &sliceTensorHandle, 1);

        diopiTensorHandle_t vIndex;
        ascend_npu::diopiIndex(ctx, &vIndex, v, &sliceTensorHandle, 1);

        AscendTensor valuesAt;
        AscendTensor qIndexAt(qIndex), kIndexAt(kIndex), vIndexAt(vIndex);
        valuesAt = torchContextAttention(ctx, qIndexAt, kIndexAt, vIndexAt, 1, reinterpret_cast<int*>(bSeqLenDataPtr)[i], head, dim);

        std::vector<AscendTensor> indices = {sliceAt};
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, out, indices, valuesAt, false, true);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
