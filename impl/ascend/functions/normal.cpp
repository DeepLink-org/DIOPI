/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    AscendTensor outAt(out);
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }

    uint64_t seed, offset;
    DIOPI_CALL(diopiGeneratorGetSeedAndOffset(generator, seed, offset));

    float meanCast = static_cast<float>(mean);
    float rstdCast = static_cast<float>(std);
    DIOPI_ASCEND_CALL_ACLNN(aclnnNormalFloatFloat, ctx, meanCast, rstdCast, seed, offset, out);
    return diopiSuccess;
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    uint64_t seed, offset;
    DIOPI_CALL(diopiGeneratorGetSeedAndOffset(generator, seed, offset));

    float meanCast = static_cast<float>(mean);
    float rstdCast = static_cast<float>(std);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceNormal, ctx, inout, meanCast, rstdCast, seed, offset);
    return diopiSuccess;
}

diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                               diopiGeneratorHandle_t generator) {
    AscendTensor outAt(out);
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }

    uint64_t seed, offset;
    DIOPI_CALL(diopiGeneratorGetSeedAndOffset(generator, seed, offset));

    DIOPI_ASCEND_CALL_ACLNN(aclnnNormalTensorTensor, ctx, mean, std, seed, offset, out);
    return diopiSuccess;
}

diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                     diopiGeneratorHandle_t generator) {
    AscendTensor outAt(out);
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }

    uint64_t seed, offset;
    DIOPI_CALL(diopiGeneratorGetSeedAndOffset(generator, seed, offset));

    float meanCast = static_cast<float>(mean);
    DIOPI_ASCEND_CALL_ACLNN(aclnnNormalFloatTensor, ctx, meanCast, std, seed, offset, out);
    return diopiSuccess;
}

diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                     diopiGeneratorHandle_t generator) {
    AscendTensor outAt(out);
    if (outAt.numel() == 0) {
        return diopiSuccess;
    }

    uint64_t seed, offset;
    DIOPI_CALL(diopiGeneratorGetSeedAndOffset(generator, seed, offset));

    float rstdCast = static_cast<float>(std);
    DIOPI_ASCEND_CALL_ACLNN(aclnnNormalTensorFloat, ctx, mean, rstdCast, seed, offset, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
