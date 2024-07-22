/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool returnCounts,
                         diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    // aclnnUnique2 only support dim == nullptr
    ASCEND_CHECK_ABORT(dim == nullptr, "dim is not supported in aclnnUnique2");

    bool returnInverse = (indices != nullptr) ? true : false;
    AscendTensor inputAt(input);
    const std::vector<int64_t>& inSizeVec = inputAt.shape();
    diopiSize_t inSize = {inSizeVec.data(), static_cast<int64_t>(inSizeVec.size())};
    std::vector<int64_t> numelSizeVec{inputAt.numel()};
    diopiSize_t numelSize = {numelSizeVec.data(), static_cast<int64_t>(numelSizeVec.size())};
    std::vector<int64_t> zeroSizeVec = {0};
    diopiSize_t zeroSize = {zeroSizeVec.data(), 1};

    // allocate temp out tensor
    diopiTensorHandle_t outTmp = nullptr;
    if (dim) {
        diopiRequireTensor(ctx, &outTmp, &inSize, nullptr, inputAt.dtype(), diopi_device);
    } else {
        diopiRequireTensor(ctx, &outTmp, &numelSize, nullptr, inputAt.dtype(), diopi_device);
    }

    // allocate temp inverse tensor
    diopiTensorHandle_t inverseTmp = nullptr;
    if (returnInverse || returnCounts) {
        diopiRequireTensor(ctx, &inverseTmp, &inSize, nullptr, diopi_dtype_int64, diopi_device);
    } else {
        diopiRequireTensor(ctx, &inverseTmp, &zeroSize, nullptr, diopi_dtype_int64, diopi_device);
    }

    // allocate temp counts tensor
    diopiTensorHandle_t countsTmp = nullptr;
    if (returnCounts) {
        diopiRequireTensor(ctx, &countsTmp, &numelSize, nullptr, diopi_dtype_int64, diopi_device);
    } else {
        diopiRequireTensor(ctx, &countsTmp, &zeroSize, nullptr, diopi_dtype_int64, diopi_device);
    }

    // call aclnnUnique2
    auto params = ::impl::ascend::aclnn_adaptor::convertParams(input, sorted, returnInverse, returnCounts, outTmp, inverseTmp, countsTmp).params();
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
    diopiSize_t outShape{viewDims, static_cast<int64_t>(viewDimNum)};
    diopiRequireTensor(ctx, out, &outShape, nullptr, inputAt.dtype(), diopi_device);
    AscendTensor outAt(*out);
    AscendTensor outTmpAt(outTmp);
    outTmpAt.view({outShape.data, outShape.data + outShape.len});
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, outTmpAt);

    // fill indices tensor
    if (returnInverse) {
        AscendTensor inverseTmpAt(inverseTmp);

        diopiSize_t inSize = {inverseTmpAt.shape().data(), static_cast<int64_t>(inverseTmpAt.shape().size())};
        AscendTensor indicesTmpAt(indices);
        if (indicesTmpAt.shape() != inverseTmpAt.shape()) {
            diopiRequireTensor(ctx, &indices, &inSize, nullptr, diopi_dtype_int64, diopi_device);
        }
        AscendTensor indicesAt(indices);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, indicesAt, inverseTmpAt);
    }

    // fill counts tensor
    if (returnCounts) {
        AscendTensor countsTmpAt(countsTmp);
        // get counts tensor shape, counts tensor is the 7th tensor in aclnnUnique2, index = 6
        constexpr int64_t countsTensorIndex = 6;
        int ret2 = aclGetViewShape(std::get<countsTensorIndex>(params), &viewDims, &viewDimNum);
        ASCEND_CHECK_ABORT(ret2 == 0, "get count aclGetViewShape failed");
        diopiSize_t countShape{viewDims, static_cast<int64_t>(viewDimNum)};
        diopiRequireTensor(ctx, counts, &countShape, nullptr, countsTmpAt.dtype(), diopi_device);
        AscendTensor countsAt(*counts);
        countsTmpAt.view({countShape.data, countShape.data + countShape.len});

        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, countsAt, countsTmpAt);
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
