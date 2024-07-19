/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    bool return_inverse = (indices != nullptr) ? true : false;
    AscendTensor inputAt(input);

    std::vector<int64_t> inSizeVec = inputAt.shape();
    diopiSize_t inSize = {inSizeVec.data(), static_cast<int64_t>(inSizeVec.size())};
    std::cout << "====inSizeVec=" << std::endl;
    for (auto i : inSizeVec) {
        std::cout << i << std::endl;
    }

    diopiTensorHandle_t outTmp = nullptr;
    std::vector<int64_t> numelVec{inputAt.numel()};
    diopiSize_t numelSize = {numelVec.data(), static_cast<int64_t>(numelVec.size())};
    if (dim) {
        diopiRequireTensor(ctx, &outTmp, &numelSize, nullptr, inputAt.dtype(), diopi_device);
    } else {
        diopiRequireTensor(ctx, &outTmp, &numelSize, nullptr, inputAt.dtype(), diopi_device);
    }

    diopiTensorHandle_t countsTmp = nullptr;
    if (dim) {
        std::vector<int64_t> inSizeVec = std::vector<int64_t>{inputAt.shape(*dim)};
        diopiSize_t inSize = {inSizeVec.data(), static_cast<int64_t>(inSizeVec.size())};
        diopiRequireTensor(ctx, &countsTmp, &inSize, nullptr, diopi_dtype_int64, diopi_device);
        if (indices == nullptr) {
            diopiRequireTensor(ctx, &indices, &inSize, nullptr, diopi_dtype_int64, diopi_device);
        }
    } else {
        diopiRequireTensor(ctx, &countsTmp, &numelSize, nullptr, diopi_dtype_int64, diopi_device);
        if (indices == nullptr) {
            diopiRequireTensor(ctx, &indices, &inSize, nullptr, diopi_dtype_int64, diopi_device);
        }
    }

    constexpr int64_t NoneN = 1000;
    int64_t dim_value = dim ? *dim : NoneN;
    auto params = ::impl::ascend::aclnn_adaptor::convertParams(input, return_inverse, return_counts, dim_value, outTmp, indices, countsTmp).params();

    // if (dim) {
    //     std::cout << "dim=" << *dim << std::endl;
    // } else {
    //     std::cout << "all dim" << std::endl;
    // }
    DIOPI_ASECND_CALL_ACLNN_TYPE_SYNC(aclnnUniqueConsecutive, ctx, params);
    std::cout << "diopiUnique finish" << std::endl;

    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
    static aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));

    // get true outShape by aclGetViewShape
    int64_t* viewDims = nullptr;
    uint64_t viewDimNum = 0;
    int ret = aclGetViewShape(std::get<4>(params), &viewDims, &viewDimNum);
    ASCEND_CHECK_ABORT(ret == 0, "out aclGetViewShape failed");

    std::cout << "viewDimNum=" << viewDimNum << std::endl;
    diopiSize_t outShape{viewDims, static_cast<int64_t>(viewDimNum)};
    std::vector<int64_t> outShapeVec(viewDims, viewDims + viewDimNum);
    std::cout << "outShapeVec=" << std::endl;
    for (auto i : outShapeVec) {
        std::cout << i << std::endl;
    }
    std::cout << "dtype=" << inputAt.dtype() << std::endl;
    // require out tensor from true outShape
    diopiRequireTensor(ctx, out, &outShape, nullptr, inputAt.dtype(), diopi_device);
    // copy outTmp to out
    AscendTensor outAt(*out);
    AscendTensor outTmpAt(outTmp);
    outTmpAt.view({outShape.data, outShape.data + outShape.len});
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, outTmpAt);

    int ret2 = aclGetViewShape(std::get<6>(params), &viewDims, &viewDimNum);
    ASCEND_CHECK_ABORT(ret2 == 0, "counts aclGetViewShape failed");
    diopiSize_t outShape1{viewDims, static_cast<int64_t>(viewDimNum)};
    // require out tensor from true outShape
    diopiRequireTensor(ctx, counts, &outShape1, nullptr, diopi_dtype_int64, diopi_device);
    // copy outTmp to out
    AscendTensor countsAt(*counts);
    AscendTensor countsTmpAt(countsTmp);
    countsTmpAt.view({outShape1.data, outShape1.data + outShape1.len});

    std::vector<int64_t> outSizeVec1(outShape1.data, outShape1.data + outShape1.len);
    std::cout << "outSizeVec1" << std::endl;
    for (auto i : outSizeVec1) {
        std::cout << i << " ";
    }
    std::cout << "outSizeVec1" << std::endl;
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, countsAt, countsTmpAt);

    if (viewDims) {
        delete viewDims;
        viewDims = nullptr;
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
