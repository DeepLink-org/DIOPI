/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>
#include <math.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/torch.h>

#include <cstring>

#ifdef USE_HIP
#include <miopen/version.h>
#endif

#define FLT_MIN __FLT_MIN__
#define FLT_MAX __FLT_MAX__

#include "../context.h"
#include "../helper.hpp"
#include "../vision_kernel.h"

namespace impl {
namespace cuda {

DIOPI_API diopiError_t diopiLmdeploySync(diopiContextHandle_t ctx) {
    impl::aten::sync(ctx);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLmdeployCopyH2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_host) {
        return diopiErrorOccurred;
    }

    impl::aten::setCurCtx(ctx);
    at::Tensor atDest = impl::aten::buildATen(dst);
    at::Tensor atSrc = impl::aten::buildATen(src);
    // Set non_blocking true to avoid stream sync thus improving performance.
    // The data is not ready when diopiCopyInp returns.
    // If you need to use it immediately, please call cudaStreamSynchronize first.
    at::native::copy_(atDest, atSrc, async);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLmdeployCopyD2H(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_host || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

    impl::aten::setCurCtx(ctx);
    at::Tensor atDest = impl::aten::buildATen(dst);
    at::Tensor atSrc = impl::aten::buildATen(src);
    // Set non_blocking true to avoid stream sync thus improving performance.
    // The data is not ready when diopiCopyInp returns.
    // If you need to use it immediately, please call cudaStreamSynchronize first.
    at::native::copy_(atDest, atSrc, async);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLmdeployCopyD2D(diopiContextHandle_t ctx, diopiTensorHandle_t dst, diopiConstTensorHandle_t src, bool async) {
    diopiDevice_t dst_dev;
    diopiGetTensorDevice(dst, &dst_dev);
    diopiDevice_t src_dev;
    diopiGetTensorDevice(src, &src_dev);
    if (dst_dev != diopiDevice_t::diopi_device || src_dev != diopiDevice_t::diopi_device) {
        return diopiErrorOccurred;
    }

    impl::aten::setCurCtx(ctx);
    at::Tensor atDest = impl::aten::buildATen(dst);
    at::Tensor atSrc = impl::aten::buildATen(src);
    // Set non_blocking true to avoid stream sync thus improving performance.
    // The data is not ready when diopiCopyInp returns.
    // If you need to use it immediately, please call cudaStreamSynchronize first.
    at::native::copy_(atDest, atSrc, async);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

}  // namespace cuda
}  // namespace impl
