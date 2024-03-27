/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cmath>

#include "../helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t expAvg, diopiTensorHandle_t expAvgSq,
                        diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay, int64_t step, bool amsgrad) {
    DIOPI_CHECK(amsgrad == false, "at present, ApplyAdamW only supports amsgrad false on ascend.");
    BEGIN_CALL_ACL_OP(input, grad, expAvg, expAvgSq, maxExpAvgSq);
    if (!inputAt.defined() || inputAt.numel() == 0) {
        return diopiSuccess;
    }

    at_npu::native::OpCommand cmd;
    // maximize is not supported in diopi for now
    bool maximize = false;
    auto dtype = inputAt.scalar_type();
    cmd.Name("ApplyAdamW")
        .Input(inputAt)
        .Input(expAvgAt)
        .Input(expAvgSqAt)
        .Input(at::Scalar(pow(beta1, step)), dtype)
        .Input(at::Scalar(pow(beta2, step)), dtype)
        .Input(at::Scalar(lr), dtype)
        .Input(at::Scalar(weightDecay), dtype)
        .Input(at::Scalar(beta1), dtype)
        .Input(at::Scalar(beta2), dtype)
        .Input(at::Scalar(eps), dtype)
        .Input(gradAt)
        .Attr<bool>("maximize", maximize)
        .Attr<bool>("amsgrad", amsgrad);  // at present, the operator supports only false.
    if (amsgrad) {
        cmd.Input(maxExpAvgSqAt);
    } else {
        cmd.Input();
    }
    cmd.Output(inputAt).Output(expAvgAt).Output(expAvgSqAt);
    cmd.Run();

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
