/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include "../common/common.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_ops_helper.hpp"

extern "C" DIOPI_API diopiError_t diopiBoxIouRotatedMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t ious, diopiConstTensorHandle_t bboxes1,
                                                         diopiConstTensorHandle_t bboxes2, int64_t mode, bool aligned) {
    mluOpHandle_t handle = impl::camb::mluOpHandlePool.get(ctx);

    impl::camb::DiopiTensor bboxes1Tensor(bboxes1);
    impl::camb::DiopiTensor bboxes2Tensor(bboxes2);
    impl::camb::DiopiTensor iousTensor(ious);

    DIOPI_CHECK(bboxes1Tensor.dtype() == diopi_dtype_float32 && bboxes2Tensor.dtype() == diopi_dtype_float32 && iousTensor.dtype() == diopi_dtype_float32,
                "mlu ops only support float32");

    impl::camb::MluOpTensorDesc bboxes1Desc(bboxes1Tensor, MLUOP_LAYOUT_ARRAY);
    impl::camb::MluOpTensorDesc bboxes2Desc(bboxes2Tensor, MLUOP_LAYOUT_ARRAY);
    impl::camb::MluOpTensorDesc iousDesc(iousTensor, MLUOP_LAYOUT_ARRAY);

    DIOPI_CALL_MLU_OP(mluOpBoxIouRotated(
        handle, mode, aligned, bboxes1Desc.get(), bboxes1Tensor.data(), bboxes2Desc.get(), bboxes2Tensor.data(), iousDesc.get(), iousTensor.data()));

    return diopiSuccess;
}
