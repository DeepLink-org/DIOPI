/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void kernelBBoxOverlaps(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dType, const void *bbox1, const void *bbox2,
                        void *ious, const int32_t numBbox1, const int32_t numBbox2, const int32_t mode, const bool aligned, const int32_t offset);
}  // namespace camb

}  // namespace impl

static void policyFunc(cnrtDim3_t *kDim, cnrtFunctionType_t *kType, const int32_t batchNumAll) {
    auto unionNum = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    auto coreDim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    auto coreNum = unionNum * coreDim;

    // Union1 policyFunc
    *kType = CNRT_FUNC_TYPE_UNION1;
    kDim->x = coreDim;
    auto needCoreNum = PAD_UP(batchNumAll, coreDim);
    kDim->y = (needCoreNum < coreNum) ? (needCoreNum / coreDim) : unionNum;
    kDim->z = 1;

    return;
}

extern "C" DIOPI_API diopiError_t diopiBboxOverlapsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t ious, diopiConstTensorHandle_t bboxes1,
                                                        diopiConstTensorHandle_t bboxes2, int64_t mode, int64_t offset, bool aligned) {
    auto bboxes1Tr = impl::camb::DiopiTensor(bboxes1);
    auto bboxes2Tr = impl::camb::DiopiTensor(bboxes2);
    auto iousTr = impl::camb::DiopiTensor(ious);

    auto rows = bboxes1Tr.size(0);
    auto cols = bboxes2Tr.size(0);
    auto batchNumAll = rows;

    if (rows * cols == 0) {
        // return if zero element
        return diopiSuccess;
    }

    // calculate task dimension
    cnrtDim3_t kDim;
    cnrtFunctionType_t kType;
    policyFunc(&kDim, &kType, batchNumAll);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    // get dtype of input
    cnrtDataType_t dType = impl::camb::dtype2CnrtDtype(bboxes1Tr.dtype());

    // launch kernel
    impl::camb::kernelBBoxOverlaps(kDim, kType, queue, dType, bboxes1Tr.data(), bboxes2Tr.data(), iousTr.data(), rows, cols, mode, aligned, offset);
    return diopiSuccess;
}
