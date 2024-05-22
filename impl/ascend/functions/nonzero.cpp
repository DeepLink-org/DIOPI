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
    int64_t inputNumEl;
    diopiGetTensorNumel(input, &inputNumEl);
    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);
    int64_t inputDim = inputSize.len;
    std::array<int64_t, 2> maxOutputSize({inputNumEl, inputDim});
    diopiSize_t outputSize = {maxOutputSize.data(), 2};

    diopiTensorHandle_t output;
    diopiRequireTensor(ctx, &output, &outputSize, nullptr, diopi_dtype_int64, diopi_device);
    auto params = DIOPI_ASECND_CALL_ACLNN_SYNC(aclnnNonzero, ctx, input, output);

    int64_t* dims = nullptr;
    uint64_t dimsNum = 0;
    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
    aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));
    aclGetViewShape(std::get<1>(params.params()), &dims, &dimsNum);

    std::vector<int64_t> outShape(dims, dims + dimsNum);
    diopiSize_t outSize = {outShape.data(), dimsNum};
    diopiRequireTensor(ctx, out, &outSize, nullptr, diopi_dtype_int64, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnSlice, ctx, output, 0, 0, dims[0], 1, *out);

    delete dims;
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
