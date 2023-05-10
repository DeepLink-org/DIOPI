/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "../mlu_helper.hpp"
#include "../diopi_helper.hpp"

namespace impl {

namespace camb {

void KernelBBoxOverlaps(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                        cnrtQueue_t queue, const cnrtDataType_t d_type,
                        const void *bbox1, const void *bbox2, void *ious,
                        const int32_t num_bbox1, const int32_t num_bbox2,
                        const int32_t mode, const bool aligned,
                        const int32_t offset);
}  // namespace camb

}  // namespace impl


static void policyFunc(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                       const int32_t batch_num_all) {
  auto union_num = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
  auto core_dim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto core_num = union_num * core_dim;

  // Union1 policyFunc
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_dim;
  auto need_core_num = PAD_UP(batch_num_all, core_dim);
  k_dim->y =
      (need_core_num < core_num) ? (need_core_num / core_dim) : union_num;
  k_dim->z = 1;

  return;
}


extern "C" DIOPI_API diopiError_t diopiBboxOverlapsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t ious_, diopiConstTensorHandle_t bboxes1_, diopiConstTensorHandle_t bboxes2_, 
                               int64_t mode, int64_t offset, bool aligned) {
    auto bboxes1 = impl::camb::DiopiTensor(bboxes1_);
    auto bboxes2 = impl::camb::DiopiTensor(bboxes2_);
    auto ious = impl::camb::DiopiTensor(ious_);

    auto rows = bboxes1.size(0);
    auto cols = bboxes2.size(0);
    auto batch_num_all = rows;

    if (rows * cols == 0) {
        // return if zero element
        return diopiSuccess;
    }

    // calculate task dimension
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(&k_dim, &k_type, batch_num_all);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    // get dtype of input
    cnrtDataType_t d_type = impl::camb::dtype2CnrtDtype(bboxes1.dtype());

    // launch kernel
    impl::camb::KernelBBoxOverlaps(k_dim, k_type, queue, d_type, bboxes1.data(), bboxes2.data(),
                       ious.data(), rows, cols, mode, aligned, offset);
    return diopiSuccess;
}
