/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
    uint64_t seed = 0;
    uint64_t offset = 0;
    diopiGeneratorGetSeedAndOffset(generator, seed, offset);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceUniform, ctx, inout, from, to, seed, offset);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
