/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                          diopiConstTensorHandle_t batch2, double beta, double alpha) {
    AscendTensor inAt(input);
    diopiScalar_t betas;
    betas.stype = inAt.dtype();
    betas.fval = beta;
    diopiScalar_t alphas;
    alphas.stype = inAt.dtype();
    alphas.fval = alpha;

    int cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnBaddbmm, ctx, input, batch1, batch2, &betas, &alphas, out, cubeMathType);
    return diopiSuccess;
}

diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta,
                             double alpha) {
    AscendTensor inAt(input);
    diopiScalar_t betas;
    betas.stype = inAt.dtype();
    betas.fval = beta;
    diopiScalar_t alphas;
    alphas.stype = inAt.dtype();
    alphas.fval = alpha;

    int cubeMathType = 0;
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceBaddbmm, ctx, input, batch1, batch2, &betas, &alphas, cubeMathType);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
