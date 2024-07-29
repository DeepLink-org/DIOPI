/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdlib>
#include <vector>

#include "../../../adaptor/csrc/impl_functions.hpp"
#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiTokenSoftmaxReduceVInference(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t logics, diopiConstTensorHandle_t v,
                                               diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                               int maxInputLen, int otherKVIndex) {
    AscendTensor outAt(out), logicsAt(logics), vAt(v), bLocAt(bLoc), bStartLocAt(bStartLoc), bSeqLenAt(bSeqLen);
    int batch = bLocAt.shape(0);
    int head = vAt.shape(1);
    int dim = vAt.shape(2);
    diopiDtype_t dtype = logicsAt.dtype();
    diopiDevice_t device = logicsAt.device();

    void* bSeqLenHost = malloc(bSeqLenAt.numel() * bSeqLenAt.elemsize());
    deviceToHost(ctx, bSeqLenAt, bSeqLenHost);
    void* bStartLocHost = malloc(bStartLocAt.numel() * bStartLocAt.elemsize());
    deviceToHost(ctx, bStartLocAt, bStartLocHost);

    int* bSeqLenAtData = reinterpret_cast<int*>(bSeqLenHost);
    int* bStartLocAtData = reinterpret_cast<int*>(bStartLocHost);

    for (int i = 0; i < batch; i++) {
        int curSeqLen = *(bSeqLenAtData + i);
        int curSeqStartLoc = *(bStartLocAtData + i);
        AscendTensor indexAt;
        makeTensor(ctx, indexAt, {curSeqLen}, diopi_dtype_int32);
        diopiScalar_t start = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc);
        diopiScalar_t end = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc + curSeqLen);
        diopiScalar_t step = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &start, &end, &step, indexAt);

        diopiTensorHandle_t indexOut;
        diopiConstTensorHandle_t indices[2] = {diopiConstTensorHandle_t(), indexAt.tensorHandle()};
        ascend_npu::diopiIndex(ctx, &indexOut, logicsAt.tensorHandle(), indices, 2);
        AscendTensor indexOutAt(indexOut);

        AscendTensor softmaxOutAt;
        makeTensor(ctx, softmaxOutAt, indexOutAt.shape(), indexOutAt.dtype());
        DIOPI_ASCEND_CALL_ACLNN(aclnnSoftmax, ctx, indexOutAt, indexOutAt.dim() - 1, softmaxOutAt);

        softmaxOutAt = softmaxOutAt.view({head, 1, 1, curSeqLen});
        AscendTensor pAt;
        makeTensor(ctx, pAt, {softmaxOutAt.shape(1), softmaxOutAt.shape(0), softmaxOutAt.shape(2), softmaxOutAt.shape(3)}, logicsAt.dtype());
        std::vector<int64_t> dims{1, 0, 2, 3};
        diopiSize_t permuteDims = vectorToDiopiSize(dims);
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, softmaxOutAt, permuteDims, pAt);

        makeTensor(ctx, indexAt, {curSeqLen}, diopi_dtype_int32);
        diopiScalar_t startVLoc = constructDiopiScalarT(diopi_dtype_int32, maxInputLen - curSeqLen);
        diopiScalar_t endVLoc = constructDiopiScalarT(diopi_dtype_int32, maxInputLen);
        diopiScalar_t stepvLoc = constructDiopiScalarT(diopi_dtype_int32, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &startVLoc, &endVLoc, &stepvLoc, indexAt);

        AscendTensor bLocAtSlice;
        makeTensor(ctx, bLocAtSlice, {1, bLocAt.shape(1)}, bLocAt.dtype());
        diopiScalar_t sliceIndexScalar = constructDiopiScalarT(diopi_dtype_int32, i);
        AscendTensor sliceIndexAt;
        makeTensorFromScalar(ctx, sliceIndexAt, &sliceIndexScalar, bLocAt.device());
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, bLocAt, (int64_t)0, sliceIndexAt, bLocAtSlice);
        bLocAtSlice.view({bLocAt.shape(1)});

        AscendTensor vLocAt;
        makeTensor(ctx, vLocAt, {curSeqLen}, bLocAt.dtype());
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexSelect, ctx, bLocAtSlice, (int64_t)0, indexAt, vLocAt);

        diopiTensorHandle_t vIndexOut;
        diopiConstTensorHandle_t indexAtHandle = vLocAt.tensorHandle();
        ascend_npu::diopiIndex(ctx, &vIndexOut, vAt.tensorHandle(), &indexAtHandle, 1);

        AscendTensor vIndexOutAt(vIndexOut);
        vIndexOutAt = vIndexOutAt.view({1, curSeqLen, head, dim});

        AscendTensor vAt;
        makeTensor(ctx, vAt, {1, head, curSeqLen, dim}, vIndexOutAt.dtype());
        dims = {0, 2, 1, 3};
        permuteDims = vectorToDiopiSize(dims);
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, vIndexOutAt, permuteDims, vAt);

        AscendTensor matmulOutAt;
        makeTensor(ctx, matmulOutAt, {pAt.shape(0), pAt.shape(1), pAt.shape(2), vAt.shape(3)}, pAt.dtype());
        DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, pAt, vAt, matmulOutAt, 0);

        diopiScalar_t scalarI = constructDiopiScalarT(diopi_dtype_int32, i);
        AscendTensor tensorI;
        makeTensorFromScalar(ctx, tensorI, &scalarI, matmulOutAt.device());
        std::vector<AscendTensor> indexPutIndices{tensorI};
        DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, outAt, indexPutIndices, matmulOutAt.view({head, dim}), false, true);
    }
    free(bSeqLenHost);
    free(bStartLocHost);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
