/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"

extern "C" {

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP();
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    // at::add_out(atOut, atInput, atOther, atAlpha);
    // op_api::add_out(atInput, atOther, atAlpha, atOut);

    acl_op::add_out(atInput, atOther, atAlpha, atOut);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP();
    return diopiAdd(ctx, input, input, other, alpha);
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP();
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);

    acl_op::add_out(atInput, at::scalar_to_tensor(atOther).to(atInput.dtype()), atAlpha, atOut);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    BEGIN_CALL_ACL_OP();
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);

    acl_op::add_(atInput, at::scalar_to_tensor(atOther).to(atInput.dtype()), atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

}  // extern "C"