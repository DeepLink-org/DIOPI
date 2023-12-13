/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

#include <ATen/ops/values_ops.h>
#include <cmath>

namespace impl {
namespace ascend {
diopiError_t diopiTokenAttentionInference(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiConstTensorHandle_t q, diopiConstTensorHandle_t k,
                                          diopiConstTensorHandle_t bLoc, diopiConstTensorHandle_t bStartLoc, diopiConstTensorHandle_t bSeqLen,
                                          int maxInputLen) {
    AscendTensor bLocAt(bLoc), qAt(q), qTmpAt, bSeqLenAt(bSeqLen), bStartLocAt(bStartLoc), kAt(k);
    int batch = bLocAt.shape(0);
    int head = qAt.shape(1);
    int dim = qAt.shape(2);

    makeTensorLike(ctx, qTmpAt, qAt);
    reshape(ctx, qAt, qTmpAt, {batch, 1, head, dim});
    auto qout = const_cast<diopiTensorHandle_t>(qTmpAt.tensorHandle());
    diopiTranspose(ctx, qout, qTmpAt.tensorHandle(), 1, 2);
    qTmpAt = AscendTensor(qout);
    auto step = constructDiopiScalarT(diopi_dtype_int32, 1);
    diopiTensorHandle_t indices, kLoc, indexOut, iTensor;


    for (int i = 0; i < batch; ++i) {
        auto iScalar = constructDiopiScalarT(diopi_dtype_int32, i);
        makeTensorFromScalar(ctx, &iScalar, &iTensor);

        int curSeqLen = bSeqLenAt[i].item<int>();       // todo wait index
        int curSeqStartLoc = bStartLocAt[i].item<int>();

        // get kLoc
        auto start = constructDiopiScalarT(diopi_dtype_int32, maxInputLen - curSeqLen);
        auto end = constructDiopiScalarT(diopi_dtype_int32, maxInputLen);
        diopiArange(ctx, indices, &start, &end, &step);
        diopiIndexSelect(ctx, kLoc, bLocAt[i], 0, indices);

        // get key
        diopiIndex(ctx, &indexOut, k, &indices, 1);
        AscendTensor indexOutAt(indexOut), keyTmp;
        indexOutAt.view({1, curSeqLen, head, dim});
        makeTensorLike(ctx, keyTmp, indexOutAt);
        diopiTensorHandle_t key;
        diopiTranspose(ctx, key, keyTmp.tensorHandle(), 1, 2);

        // get outLoc
        auto start1 = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc);
        auto end1 = constructDiopiScalarT(diopi_dtype_int32, curSeqStartLoc + curSeqLen);
        diopiTensorHandle_t outLoc;
        diopiArange(ctx, outLoc, &start1, &end1, &step);

        // get values
        diopiTensorHandle_t d, indexOut1, keyOut, mulOut, divOut;
        diopiIndex(ctx, &indexOut1, q, &iTensor, 1);
        diopiTranspose(ctx, keyOut, key, 2, 3);
        diopiMatmul(ctx, mulOut, indexOut1, keyOut);

        auto dNum = std::sqrt(dim);
        auto dScalar = constructDiopiScalarT(diopi_dtype_float64, dNum);
        makeTensorFromScalar(ctx, &dScalar, &d);

        diopiDiv(ctx, divOut, mulOut, d, diopiRoundMode_t::RoundModeNone);
        AscendTensor dOutAt(divOut);
        dOutAt.view({head, curSeqLen});
        AscendTensor valuesAt;
        makeTensorLike(ctx, valuesAt, dOutAt);

        // fill data
        AscendTensor outLocAt(outLoc);
        int firstDim = outLocAt.shape(0);
        auto start2 = constructDiopiScalarT(diopi_dtype_int32, 0);
        auto end2 = constructDiopiScalarT(diopi_dtype_int32, firstDim);
        diopiTensorHandle_t firstTensor, indexOut2;
        diopiArange(ctx, firstTensor, &start2, &end2, &step);
        diopiStack(ctx, indexOut2, &firstTensor, firstDim, 1);
        diopiIndexPutInp(ctx, attentionOut, valuesAt.tensorHandle(), indexOut2, outLocAt.shape(1), true);

        return diopiSuccess;
    }
}

}  // namespace ascend
}  // namespace impl
