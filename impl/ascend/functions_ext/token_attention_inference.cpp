/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdlib>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../../../adaptor/csrc/impl_functions.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    AscendTensor attentionOutAt(attentionOut), qAt(q), kAt(k), bLocAt(bLoc), bStartLocAt(bStartLoc), bSeqLenAt(bSeqLen);
    int batch = bLocAt.shape(0);
    int head = qAt.shape(1);
    int dim = qAt.shape(2);
    qAt = qAt.view({batch, head, 1, dim});
    diopiDtype_t dtype = qAt.dtype();
    diopiDevice_t device = qAt.device();

    void* bSeqLenHost = malloc(bSeqLenAt.numel() * bSeqLenAt.elemsize());
    deviceToHost(ctx, bSeqLenAt, bSeqLenHost);
    void* bStartLocHost = malloc(bStartLocAt.numel() * bStartLocAt.elemsize());
    deviceToHost(ctx, bStartLocAt, bStartLocHost);

    int* bSeqLenAtData = reinterpret_cast<int*>(bSeqLenHost);
    int* bStartLocAtData = reinterpret_cast<int*>(bStartLocHost);

    for (int i = 0; i < batch; i++) {
        int curSeqLen = *(bSeqLenAtData + i);
        int curSeqStartLoc = *(bStartLocAtData + i);
        AscendTensor kLocAt, indexAt;
        makeTensor(ctx, indexAt, {curSeqLen}, diopi_dtype_int32);
        diopiScalar_t start = constructDiopiScalarT(diopi_dtype_int32, maxInputLen - curSeqLen);
        diopiScalar_t end = constructDiopiScalarT(diopi_dtype_int32, maxInputLen);
        diopiScalar_t step = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &start, &end, &step, indexAt);

        AscendTensor bLocAtSlice;
        makeTensor(ctx, bLocAtSlice, {1, bLocAt.shape(1)}, bLocAt.dtype());

        diopiScalar_t sliceIndexScalar = constructDiopiScalarT(diopi_dtype_int32, i);
        AscendTensor sliceIndexAt;
        makeTensorFromScalar(ctx, sliceIndexAt, &sliceIndexScalar, bLocAt.device());
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, bLocAt, (int64_t)0, sliceIndexAt, bLocAtSlice);
        bLocAtSlice.view({bLocAt.shape(1)});
        makeTensor(ctx, kLocAt, {curSeqLen}, bLocAt.dtype());
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, bLocAtSlice, (int64_t)0, indexAt, kLocAt);

        diopiTensorHandle_t keyTmp;
        diopiConstTensorHandle_t indexAtHandle = kLocAt.tensorHandle();
        ascend_npu::diopiIndex(ctx, &keyTmp, k, &indexAtHandle, 1);

        AscendTensor keyTmpAt(keyTmp);

        keyTmpAt = keyTmpAt.unsqueeze(0);
        AscendTensor keyAt;
        makeTensor(ctx, keyAt, {1, head, curSeqLen, dim}, keyTmpAt.dtype());
        std::vector<int64_t> dims{0, 2, 1, 3};
        diopiSize_t permuteDims = vectorToDiopiSize(dims);
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, keyTmpAt, permuteDims, keyAt);

        AscendTensor outLocAt;
        makeTensor(ctx, outLocAt, {curSeqLen}, diopi_dtype_int32);
        diopiScalar_t startScalar = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc);
        diopiScalar_t endScalar = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc + curSeqLen);
        diopiScalar_t stepScalar = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &startScalar, &endScalar, &stepScalar, outLocAt);

        AscendTensor scalarTensor;
        diopiScalar_t scalarI = constructDiopiScalarT(diopi_dtype_int64, i);
        makeTensorFromScalar(ctx, scalarTensor, &scalarI, qAt.device());

        diopiTensorHandle_t qIndex;
        diopiConstTensorHandle_t scalarTensorHandle = scalarTensor.tensorHandle();
        ascend_npu::diopiIndex(ctx, &qIndex, qAt.tensorHandle(), &scalarTensorHandle , 1);

        AscendTensor qIndexAt(qIndex);

        AscendTensor matmulOutAt;
        makeTensor(ctx, matmulOutAt, {keyAt.shape(0), keyAt.shape(1), qIndexAt.shape(0), keyAt.shape(2)}, keyAt.dtype());
        qIndexAt.unsqueeze(0);

        AscendTensor keyTmp2At;
        makeTensor(ctx, keyTmp2At, {keyAt.shape(0), keyAt.shape(1), keyAt.shape(3), keyAt.shape(2)}, keyAt.dtype());
        dims = {0, 1, 3, 2};
        permuteDims = vectorToDiopiSize(dims);
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, keyAt, permuteDims, keyTmp2At);

        DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, qIndexAt.view({qIndexAt.shape(0), qIndexAt.shape(2), qIndexAt.shape(1), qIndexAt.shape(3)}), keyTmp2At, matmulOutAt, 0);

        AscendTensor sqrtDimAt;
        diopiScalar_t sqrtDim = constructDiopiScalarT(qAt.dtype(), sqrt(dim));
        makeTensorFromScalar(ctx, sqrtDimAt, &sqrtDim, matmulOutAt.device());
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceDiv, ctx, matmulOutAt, sqrtDimAt);

        std::vector<AscendTensor> indices{AscendTensor(), outLocAt};
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, attentionOutAt, indices, matmulOutAt.view({head, curSeqLen}), false, true);
    }
    free(bSeqLenHost);
    free(bStartLocHost);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
