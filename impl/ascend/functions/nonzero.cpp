/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    // In the case where all elements of the input are non-zero, calculate the maximum out size.
    AscendTensor inputAt(input);
    int64_t maxOutSizeData[2]{inputAt.numel(), inputAt.dim()};
    const int64_t outLen = 2;
    diopiSize_t maxOutSize = {maxOutSizeData, outLen};

    // build outTmp with maxOutSize and call aclnnNonZero to update outTmp
    diopiTensorHandle_t outTmp;
    diopiRequireTensor(ctx, &outTmp, &maxOutSize, nullptr, diopi_dtype_int64, diopi_device);
    auto params = aclnn_adaptor::convertParams(input, outTmp).params();
    DIOPI_ASECND_CALL_ACLNN_TYPE_SYNC(aclnnNonzero, ctx, params);

    // get the true out Shape
    int64_t* dims = nullptr;
    uint64_t dimsNum = 0;
    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
    aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));
    aclGetViewShape(std::get<1>(params), &dims, &dimsNum);
    std::vector<int64_t> outShape(dims, dims + dimsNum);
    diopiSize_t outSize = {outShape.data(), static_cast<int64_t>(dimsNum)};

    // copy outTmp to out
    diopiRequireTensor(ctx, out, &outSize, nullptr, diopi_dtype_int64, diopi_device);
    AscendTensor outTmpAt(outTmp);
    AscendTensor outAt(*out);
    outTmpAt.view(outShape);
    outAt.view(outShape);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, outTmpAt);

    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
