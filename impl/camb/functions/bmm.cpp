/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor mat1Tensor = DiopiTensor(input);
    DiopiTensor mat2Tensor = DiopiTensor(mat2);
    DiopiTensor outputTensor = DiopiTensor(out);

    DiopiTensor mat1Casted = mat1Tensor;
    DiopiTensor mat2Casted = mat2Tensor;
    DiopiTensor outputCasted = outputTensor;

    std::vector<DiopiTensor*> tensors{&mat1Casted, &mat2Casted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    CnnlTensorDesc mat1Desc(mat1Casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat2Desc(mat2Casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outputCasted, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(
        cnnlBatchMatMul(handle, false, false, mat1Desc.get(), mat1Casted.data(), mat2Desc.get(), mat2Casted.data(), outDesc.get(), outputCasted.data()));
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
