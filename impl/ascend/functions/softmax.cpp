#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<1, 1>("SoftmaxV2").addInput(input, ACL_FORMAT_ND).setAttr<int64_t>("axes", dim_list).addOutput(out, ACL_FORMAT_ND).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                       diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<2, 1>("SoftmaxGrad")
        .addInput(grad_output, ACL_FORMAT_ND)
        .addInput(output, ACL_FORMAT_ND)
        .setAttr<int64_t>("axes", dim_list)
        .addOutput(grad_input, ACL_FORMAT_ND)
        .run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<1, 1>("LogSoftmaxV2").addInput(input, ACL_FORMAT_ND).setAttr("axes", dim_list).addOutput(out, ACL_FORMAT_ND).run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                          diopiConstTensorHandle_t output, int64_t dim) {
    std::vector<int64_t> dim_list = {dim};
    AclOpRunner<2, 1>("LogSoftmaxGrad")
        .addInput(grad_output, ACL_FORMAT_ND)
        .addInput(output, ACL_FORMAT_ND)
        .setAttr("axes", dim_list)
        .addOutput(grad_input, ACL_FORMAT_ND)
        .run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
