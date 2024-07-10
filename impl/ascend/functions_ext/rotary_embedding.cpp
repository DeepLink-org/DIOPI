/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
namespace {

std::vector<int64_t> expandTo4DShape(const std::vector<int64_t>& shape) {
    std::size_t inputSize = shape.size();
    if (shape.size() == 4) {
        return shape;
    }

    static const int64_t tDim = 4;
    std::vector<int64_t> expShape(tDim, 1);
    for (int i = 0; i < inputSize; i++) {
        expShape[i + tDim - inputSize] = shape[i];
    }
    return expShape;
}

}  // namespace

DIOPI_API diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                            diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    if (interleaved) {
        // warning(const char* file, int lineNum, const char* funcName, const char* format, ...)
        warning(__FILE__, __LINE__, "diopiRotaryEmbedding", "interleaved rotary embedding is not supported yet");
        return diopiNoImplement;
    }

    AscendTensor xAt(x);
    AscendTensor cosAt(cos);
    AscendTensor sinAt(sin);
    AscendTensor outAt(out);
    ASCEND_CHECK_ABORT(xAt.shape(-1) == 2 * cosAt.shape(-1) && xAt.shape(-1) == 2 * sinAt.shape(-1),
                       "The size of the last dimension of x must be twice the size of the corresponding dimensions of cos and sin!");
    if (xAt.numel() == 0) {
        return diopiSuccess;
    }

    if (xAt.dim() >= 5) {
        warning(__FILE__, __LINE__, "diopiRotaryEmbedding", "rotary embedding not support 5D tensor yet");
        return diopi5DNotSupported;
    }

    AscendTensor xView = xAt.view(expandTo4DShape(xAt.shape()));
    AscendTensor outView = outAt.view(expandTo4DShape(outAt.shape()));
    AscendTensor cosView = cosAt.view(expandTo4DShape(cosAt.shape()));
    AscendTensor sinView = sinAt.view(expandTo4DShape(sinAt.shape()));

    diopiTensorHandle_t cosCat;
    std::vector<int64_t> cosCatShape(cosView.shape());
    cosCatShape[cosCatShape.size() - 1] *= 2;
    diopiSize_t cosDiopiSize = vectorToDiopiSize(cosCatShape);
    diopiRequireTensor(ctx, &cosCat, &cosDiopiSize, nullptr, cosView.dtype(), diopi_device);
    std::vector<AscendTensor> cosTensorVec{cosView, cosView};
    DIOPI_ASCEND_CALL_ACLNN(aclnnCat, ctx, cosTensorVec, cosCatShape.size() - 1, cosCat);

    diopiTensorHandle_t sinCat;
    std::vector<int64_t> sinCatShape(sinView.shape());
    sinCatShape[sinCatShape.size() - 1] *= 2;
    diopiSize_t sinDiopiSize = vectorToDiopiSize(sinCatShape);
    diopiRequireTensor(ctx, &sinCat, &sinDiopiSize, nullptr, sinView.dtype(), diopi_device);
    std::vector<AscendTensor> sinTensorVec{sinView, sinView};
    DIOPI_ASCEND_CALL_ACLNN(aclnnCat, ctx, sinTensorVec, sinCatShape.size() - 1, sinCat);

    if (conj) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNeg, ctx, sinCat);
    }

    std::vector<int64_t> xViewShape = xView.shape();
    uint64_t splitSections = xViewShape[xViewShape.size() - 1] / 2;
    int64_t dim = xViewShape.size() - 1;
    std::vector<int64_t> chunkSize(xViewShape);
    chunkSize[dim] = splitSections;
    diopiSize_t xViewChunkShape = vectorToDiopiSize(chunkSize);

    diopiTensorHandle_t xViewChunk1;
    diopiRequireTensor(ctx, &xViewChunk1, &xViewChunkShape, nullptr, xView.dtype(), diopi_device);
    diopiTensorHandle_t xViewChunk2;
    diopiRequireTensor(ctx, &xViewChunk2, &xViewChunkShape, nullptr, xView.dtype(), diopi_device);
    std::vector<diopiTensorHandle_t> chunkTensors{xViewChunk1, xViewChunk2};

    DIOPI_ASCEND_CALL_ACLNN(aclnnSplitTensor, ctx, xView, splitSections, dim, chunkTensors);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNeg, ctx, chunkTensors[1]);
    std::vector<diopiConstTensorHandle_t> xChunk{xViewChunk2, xViewChunk1};

    diopiTensorHandle_t xNew;
    std::vector<int64_t> xNewShape = xView.shape();
    diopiSize_t xNewSize = vectorToDiopiSize(xNewShape);
    diopiRequireTensor(ctx, &xNew, &xNewSize, nullptr, xView.dtype(), diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnCat, ctx, xChunk, xNewShape.size() - 1, xNew);

    std::vector<int64_t> outSize = outView.shape();
    diopiSize_t outShape = vectorToDiopiSize(outSize);
    diopiTensorHandle_t cosX;
    diopiRequireTensor(ctx, &cosX, &outShape, nullptr, outView.dtype(), diopi_device);
    diopiTensorHandle_t sinX;
    diopiRequireTensor(ctx, &sinX, &outShape, nullptr, outView.dtype(), diopi_device);

    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, cosCat, xView, cosX);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, sinCat, xNew, sinX);

    // diopiScalar_t alpha = constructDiopiScalarT(outView.dtype(), 1.0);
    // alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
    diopiScalar_t alpha = constructDiopiScalarT(diopi_dtype_float32, 1.0);
    DIOPI_ASCEND_CALL_ACLNN(aclnnAdd, ctx, cosX, sinX, &alpha, outView);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
