/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include <cmath>
#include <memory>

#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void kernelDynamicVoxelize(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const void *points, void *coors, const float voxelX,
                           const float voxelY, const float voxelZ, const float coorsXMin, const float coorsYMin, const float coorsZMin, const float coorsXMax,
                           const float coorsYMax, const float coorsZMax, const int32_t gridX, const int32_t gridY, const int32_t gridZ, const int32_t numPoints,
                           const int32_t numFeatures);

void kernelPoint2Voxel(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, void *coors, void *pointToPointidx, void *pointToVoxelidx,
                       const int32_t numPoints, const int32_t maxPoints);

void kernelCalcPointsPerVoxel(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, void *pointToPointidx, void *pointToVoxelidx, void *coorToVoxelidx,
                              void *numPointsPerVoxel, void *voxelNum, const int32_t maxVoxels, const int32_t numPoints);

void kernelAssignVoxelsCoors(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const void *points, void *tempCoors, void *pointToVoxelidx,
                             void *coorToVoxelidx, void *voxels, void *coors, const int32_t maxPoints, const int32_t numPoints, const int32_t numFeatures);
}  // namespace camb

}  // namespace impl

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// policy function
static void policyFuncDefault(cnrtDim3_t *kDim, cnrtFunctionType_t *kType, const int numPoints) {
    kDim->x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    kDim->y = MIN((numPoints + kDim->x - 1) / kDim->x, impl::camb::getDeviceAttr(cnrtAttrClusterCount));
    kDim->z = 1;
    *kType = CNRT_FUNC_TYPE_UNION1;
}

// policy function
static void policyFuncCalcPointsPerVoxel(cnrtDim3_t *kDim, cnrtFunctionType_t *kType, const int numPoints) {
    kDim->x = 1;
    kDim->y = 1;
    kDim->z = 1;
    *kType = CNRT_FUNC_TYPE_BLOCK;
}

extern "C" diopiError_t diopiHardVoxelizeMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t voxelsTr, diopiTensorHandle_t coorsTr,
                                              diopiTensorHandle_t numPointsPerVoxelTr, diopiConstTensorHandle_t pointsTr, int64_t *voxelNumTr,
                                              const float *voxelSizeTr, int64_t voxelSizeLen, const float *coorsRangeTr, int64_t coorsRangeLen,
                                              int64_t maxPoints, int64_t maxVoxels, int64_t nDim, bool deterministic) {
    auto voxels = impl::camb::DiopiTensor(voxelsTr);
    auto coors = impl::camb::DiopiTensor(coorsTr);
    auto numPointsPerVoxel = impl::camb::DiopiTensor(numPointsPerVoxelTr);
    auto points = impl::camb::DiopiTensor(pointsTr);

    const int numPoints = points.size(0);
    const int numFeatures = points.size(1);

    std::vector<float> voxelSizeV(voxelSizeTr, voxelSizeTr + voxelSizeLen);
    std::vector<float> coorsRangeV(coorsRangeTr, coorsRangeTr + coorsRangeLen);

    // check zero element
    if (maxPoints == 0 || maxVoxels == 0) {
        *voxelNumTr = 0;
        return diopiSuccess;
    }

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    // calculate task dimension
    cnrtDim3_t kDim;
    cnrtFunctionType_t kType;
    policyFuncDefault(&kDim, &kType, numPoints);

    // 1. link point to corresponding voxel coors
    const float voxelX = voxelSizeV[0];
    const float voxelY = voxelSizeV[1];
    const float voxelZ = voxelSizeV[2];
    const float coorsXMin = coorsRangeV[0];
    const float coorsYMin = coorsRangeV[1];
    const float coorsZMin = coorsRangeV[2];
    const float coorsXMax = coorsRangeV[3];
    const float coorsYMax = coorsRangeV[4];
    const float coorsZMax = coorsRangeV[5];

    const int gridX = std::round((coorsXMax - coorsXMin) / voxelX);
    const int gridY = std::round((coorsYMax - coorsYMin) / voxelY);
    const int gridZ = std::round((coorsZMax - coorsZMin) / voxelZ);

    diopiScalar_t scalar = impl::camb::constructDiopiScalarT(diopi_dtype_int32, 0);
    auto tempCoors = impl::camb::requiresTensor(ctx, {nDim, numPoints}, diopi_dtype_int32);
    DIOPI_CALL(impl::camb::diopiFill(ctx, diopiTensorHandle_t(tempCoors), &scalar));

    impl::camb::kernelDynamicVoxelize(kDim,
                                      kType,
                                      queue,
                                      points.data(),
                                      tempCoors.data(),
                                      voxelX,
                                      voxelY,
                                      voxelZ,
                                      coorsXMin,
                                      coorsYMin,
                                      coorsZMin,
                                      coorsXMax,
                                      coorsYMax,
                                      coorsZMax,
                                      gridX,
                                      gridY,
                                      gridZ,
                                      numPoints,
                                      numFeatures);

    // 2. map point to the idx of the corresponding voxel, find duplicate coor
    auto pointToPointidx = impl::camb::requiresTensor(ctx, {numPoints}, diopi_dtype_int32);
    DIOPI_CALL(impl::camb::diopiFill(ctx, diopiTensorHandle_t(pointToPointidx), &scalar));

    auto pointToVoxelidx = impl::camb::requiresTensor(ctx, {numPoints}, diopi_dtype_int32);
    DIOPI_CALL(impl::camb::diopiFill(ctx, diopiTensorHandle_t(pointToVoxelidx), &scalar));

    impl::camb::kernelPoint2Voxel(kDim, kType, queue, tempCoors.data(), pointToPointidx.data(), pointToVoxelidx.data(), numPoints, maxPoints);

    // calculate task dimension
    cnrtDim3_t kDimCalcPointsPerVoxel;
    cnrtFunctionType_t kTypeCalcPointsPerVoxel;
    policyFuncCalcPointsPerVoxel(&kDimCalcPointsPerVoxel, &kTypeCalcPointsPerVoxel, numPoints);

    // 3. determine voxel num and voxel's coor index
    auto coorToVoxelidx = impl::camb::requiresTensor(ctx, {numPoints}, diopi_dtype_int32);
    DIOPI_CALL(impl::camb::diopiFill(ctx, diopiTensorHandle_t(coorToVoxelidx), &scalar));
    auto voxelNumTrtmp = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int32);
    DIOPI_CALL(impl::camb::diopiFill(ctx, diopiTensorHandle_t(voxelNumTrtmp), &scalar));

    impl::camb::kernelCalcPointsPerVoxel(kDimCalcPointsPerVoxel,
                                         kTypeCalcPointsPerVoxel,
                                         queue,
                                         pointToPointidx.data(),
                                         pointToVoxelidx.data(),
                                         coorToVoxelidx.data(),
                                         numPointsPerVoxel.data(),
                                         voxelNumTrtmp.data(),
                                         maxVoxels,
                                         numPoints);

    // 4. copy point features and coors of each voxels to voxels
    impl::camb::kernelAssignVoxelsCoors(kDim,
                                        kType,
                                        queue,
                                        points.data(),
                                        tempCoors.data(),
                                        pointToVoxelidx.data(),
                                        coorToVoxelidx.data(),
                                        voxels.data(),
                                        coors.data(),
                                        maxPoints,
                                        numPoints,
                                        numFeatures);
    int32_t voxelNumTrcpu;
    cnrtMemcpyAsync(&voxelNumTrcpu, voxelNumTrtmp.data(), sizeof(int32_t), impl::camb::getStream(ctx), cnrtMemcpyDevToHost);
    impl::camb::syncStreamInCtx(ctx);
    *voxelNumTr = voxelNumTrcpu;
    return diopiSuccess;
}
