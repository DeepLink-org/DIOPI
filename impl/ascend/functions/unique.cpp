/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <vector>

#include "../aclnn/adaptor.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool returnCounts,
                         diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    // aclnnUnique2 only supports when dim is nullptr. If dim is not nullptr, aclnnUniqueDim should be used.
    ASCEND_CHECK_ABORT(dim == nullptr, "dim is not supported in aclnnUnique2");

    // allocate temp out tensor
    diopiTensorHandle_t outTmp = nullptr;
    AscendTensor inputAt(input), outTmpAt(outTmp);
    if (dim) {
        ASCEND_CHECK_ABORT(false, "dim is not supported in aclnnUnique2, need use aclnnUniqueDim.");
    } else {
        makeTensor(ctx, outTmpAt, {inputAt.numel()}, inputAt.dtype());
    }

    // allocate temp inverse tensor
    diopiTensorHandle_t inverseTmp = nullptr;
    AscendTensor inverseTmpAt(inverseTmp);
    bool returnInverse = (indices != nullptr) ? true : false;
    std::vector<int64_t> zeroShape = {0};
    if (returnInverse || returnCounts) {
        makeTensor(ctx, inverseTmpAt, inputAt.shape(), diopi_dtype_int64);
    } else {
        makeTensor(ctx, inverseTmpAt, zeroShape, diopi_dtype_int64);
    }

    // allocate temp counts tensor
    diopiTensorHandle_t countsTmp = nullptr;
    AscendTensor countsTmpAt(countsTmp);
    if (returnCounts) {
        makeTensor(ctx, countsTmpAt, {inputAt.numel()}, diopi_dtype_int64);
    } else {
        makeTensor(ctx, countsTmpAt, zeroShape, diopi_dtype_int64);
    }

    // call aclnnUnique2
    auto params = ::impl::ascend::aclnn_adaptor::convertParams(input, sorted, returnInverse, returnCounts, outTmpAt, inverseTmpAt, countsTmpAt).params();
    DIOPI_ASECND_CALL_ACLNN_TYPE_SYNC(aclnnUnique2, ctx, params);

    // get true outShape by aclGetViewShape
    int64_t* viewDims = nullptr;
    uint64_t viewDimNum = 0;
    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
    static aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));
    // get out tensor shape, out tensor is the 5th tensor in aclnnUnique2, index = 4
    constexpr int64_t outputTensorIndex = 4;
    int ret = aclGetViewShape(std::get<outputTensorIndex>(params), &viewDims, &viewDimNum);
    ASCEND_CHECK_ABORT(ret == 0, "get out aclGetViewShape failed");

    // fill out tensor
    AscendTensor outReshapeAt = reshape(ctx, outTmpAt, {viewDims, viewDims + viewDimNum});
    *out = const_cast<diopiTensorHandle_t>(outReshapeAt.tensorHandle());

    // fill indices tensor
    if (returnInverse) {
        indices = const_cast<diopiTensorHandle_t>(inverseTmpAt.tensorHandle());
    }

    // fill counts tensor
    if (returnCounts) {
        // get counts tensor shape, counts tensor is the 7th tensor in aclnnUnique2, index = 6
        constexpr int64_t countsTensorIndex = 6;
        int ret2 = aclGetViewShape(std::get<countsTensorIndex>(params), &viewDims, &viewDimNum);
        ASCEND_CHECK_ABORT(ret2 == 0, "get count aclGetViewShape failed");

        AscendTensor countsReshapeAt = reshape(ctx, countsTmpAt, {viewDims, viewDims + viewDimNum});
        *counts = const_cast<diopiTensorHandle_t>(countsReshapeAt.tensorHandle());
    }

    // delete viewDims pointer
    if (viewDims) {
        delete viewDims;
        viewDims = nullptr;
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
