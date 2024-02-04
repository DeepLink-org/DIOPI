/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../aclnn/aclnn.hpp"
#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiCos(ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    aclnnCosAdaptor(ctx, input, out);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
