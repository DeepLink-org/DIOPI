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

extern "C" {

#define DIOPI_CHECK(expr)                                           \
    do {                                                            \
        diopiError_t ret = expr;                                    \
        if (ret != diopiSuccess) {                                  \
            printf(#expr " error at %s:%d.\n", __FILE__, __LINE__); \
            return ret;                                             \
        }                                                           \
    } while (false);

#define DIOPI_CHECK_FMT(expr, fmt, args...)                          \
    do {                                                             \
        diopiError_t ret = expr;                                     \
        if (ret != diopiSuccess) {                                   \
            printf(#fmt " at %s:%d.\n", ##args, __FILE__, __LINE__); \
            return ret;                                              \
        }                                                            \
    } while (false);

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

DIOPI_API diopiError_t diopiFusedSiluFfnInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t weight1,
                                            diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3, diopiTensorHandle_t workspace,
                                            int64_t* workspace_size, int64_t fusion_level) {
    if (fusion_level >= 0) {
        diopiSize_t shapeinfo;
        diopiGetTensorShape(inoutput, &shapeinfo);
        int64_t token_num = shapeinfo.data[0];
        diopiGetTensorShape(weight1, &shapeinfo);
        int64_t inter_size = shapeinfo.data[1];
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = 2 * itemsize * token_num * inter_size;
            return diopiSuccess;
        }
        void* dataptr;
        diopiGetTensorData(workspace, &dataptr);
        diopiDevice_t device;
        diopiGetTensorDevice(workspace, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(workspace, &dtype);
        std::vector<int64_t> shape(2);
        diopiSize_t newshape{shape.data(), 2};
        shape[0] = token_num;
        shape[1] = inter_size;
        diopiSize_t strideW1{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(dataptr)), -1};
        diopiSize_t strideW3{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(dataptr) + itemsize * token_num * inter_size)), -1};
        diopiTensorHandle_t matmulW1;
        diopiTensorHandle_t matmulW3;
        diopiRequireTensor(ctx, &matmulW1, &newshape, &strideW1, dtype, device);
        diopiRequireTensor(ctx, &matmulW3, &newshape, &strideW3, dtype, device);

        DIOPI_CHECK(diopiMm(ctx, matmulW1, inoutput, weight1));
        DIOPI_CHECK(diopiMm(ctx, matmulW3, inoutput, weight3));
        DIOPI_CHECK(diopiSiluInp(ctx, matmulW1));
        DIOPI_CHECK(diopiMulInp(ctx, matmulW1, matmulW3));
        DIOPI_CHECK(diopiMm(ctx, inoutput, matmulW1, weight2));
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

}  // extern "C"
