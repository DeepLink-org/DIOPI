/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

static const int64_t bitNumber = 128;
static const int64_t uInt8BitNumber = 8;

diopiError_t npuDropoutOut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                           diopiGeneratorHandle_t generator) {
    AscendTensor inAt(input);
    int64_t length = (inAt.numel() + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
    std::vector<int64_t> szVec(1, length);
    diopiSize_t maskSize = vectorToDiopiSize(szVec);
    diopiTensorHandle_t maskNpu;
    diopiError_t ret = diopiRequireTensor(ctx, &maskNpu, &maskSize, nullptr, diopi_dtype_uint8, diopi_device);
    ASCEND_CHECK_ABORT(ret == diopiSuccess, "[npuDropoutOut] require tensor for mask failed.");

    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, 10);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;

    DIOPI_ASCEND_CALL_ACLNN(aclnnDropoutGenMask, ctx, inAt.shape(), p, seed, offset, maskNpu);
    DIOPI_ASCEND_CALL_ACLNN(aclnnDropoutDoMask, ctx, input, maskNpu, p, out);

    diopiScalar_t ref = constructDiopiScalarT(inAt.dtype(), 0.0);
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeScalar, ctx, out, &ref, mask);

    return diopiSuccess;
}

diopiError_t npuDropout2dOut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                             diopiGeneratorHandle_t generator) {
    AscendTensor inputAt(input);
    AscendTensor maskAt(mask);
    std::vector<int64_t> maskShape(maskAt.shape());
    const diopiSize_t maskDiopiSize = vectorToDiopiSize(maskShape);

    diopiTensorHandle_t input2d;
    diopiRequireTensor(ctx, &input2d, &maskDiopiSize, nullptr, inputAt.dtype(), diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, input2d);
    diopiTensorHandle_t out2d;
    diopiRequireTensor(ctx, &out2d, &maskDiopiSize, nullptr, inputAt.dtype(), diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, out2d);

    AscendTensor inAt(input2d);
    int64_t length = (inAt.numel() + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
    std::vector<int64_t> szVec(1, length);
    diopiSize_t maskNpuSize = vectorToDiopiSize(szVec);
    diopiTensorHandle_t maskNpu;
    diopiError_t ret = diopiRequireTensor(ctx, &maskNpu, &maskNpuSize, nullptr, diopi_dtype_uint8, diopi_device);
    ASCEND_CHECK_ABORT(ret == diopiSuccess, "[npuDropout2dOut] require tensor for mask failed.");

    const std::pair<uint64_t, uint64_t> gen = getSeedAndOffset(ctx, generator, 10);
    const uint64_t seed = gen.first;
    const uint64_t offset = gen.second;

    DIOPI_ASCEND_CALL_ACLNN(aclnnDropoutGenMask, ctx, inAt.shape(), p, seed, offset, maskNpu);
    DIOPI_ASCEND_CALL_ACLNN(aclnnDropoutDoMask, ctx, input2d, maskNpu, p, out2d);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMul, ctx, input, out2d, out);

    diopiScalar_t ref = constructDiopiScalarT(inputAt.dtype(), 0.0);
    DIOPI_ASCEND_CALL_ACLNN(aclnnNeScalar, ctx, out2d, &ref, mask);

    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train,
                          diopiGeneratorHandle_t generator) {
    if (p == 0 || train == false) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, out, input);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, mask);
        return diopiSuccess;
    }

    if (p == 1) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, input);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, mask);
        return diopiSuccess;
    }

    AscendTensor inAt(input);
    AscendTensor maskAt(mask);
    if (inAt.shape() != maskAt.shape()) {
        npuDropout2dOut(ctx, out, mask, input, p, generator);
    } else {
        npuDropoutOut(ctx, out, mask, input, p, generator);
    }

    return diopiSuccess;
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    if (p == 0 || train == false) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, mask);
        return diopiSuccess;
    }

    if (p == 1) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, input);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, mask);
        return diopiSuccess;
    }

    AscendTensor inAt(input);
    AscendTensor maskAt(mask);
    if (inAt.shape() != maskAt.shape()) {
        npuDropout2dOut(ctx, input, mask, input, p, generator);
    } else {
        npuDropoutOut(ctx, input, mask, input, p, generator);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
