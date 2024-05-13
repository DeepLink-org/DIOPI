/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"
namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {    
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (inputNumel == 0) {
        return diopiSuccess;
    }

    bool exclusive = false;
    bool reverse = false;
    DIOPI_ASCEND_CALL_ACLNN(aclnnCumsumV2, ctx, input, dim, exclusive, reverse, out);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
