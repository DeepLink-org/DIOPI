/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>

#include "helper.hpp"


static const char* name = "MuxiDevice";
const char* diopiGetVendorName() { return name; }

namespace impl {
namespace muxi {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    impl::aten::setCurStream(ctx);
    DIOPI_CHECK_PTR(tensors);
    auto tensorList = impl::aten::buildATenList(tensors, insNum);
    auto atOut = impl::aten::buildATen(out);
    at::cat_out(atOut, tensorList, dim);

    return diopiSuccess;
}

}  // namespace muxi
}  // namespace impl
