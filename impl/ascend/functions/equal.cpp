/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiEqual(diopiContextHandle_t ctx, bool* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    // make outAt and call aclnnEqual
    AscendTensor outAt;
    diopiScalar_t boolScalar = constructDiopiScalarT(diopi_dtype_bool, false);
    makeTensorFromScalar(ctx, outAt, &boolScalar);
    DIOPI_ASCEND_CALL_ACLNN(aclnnEqual, ctx, input, other, outAt);

    // write data of outAt to out
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    CALL_ACLRT(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)));
    CALL_ACLRT(aclrtMemcpy(out, sizeof(bool), outAt.data(), sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST));
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
