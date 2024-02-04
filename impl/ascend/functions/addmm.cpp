/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    AscendTensor inputTr(input);
    AscendTensor mat1Tr(mat1);
    AscendTensor mat2Tr(mat2);

    // calculate (mat1 @ mat2)
    std::vector<int64_t> tempShape(2);
    tempShape[0] = mat1Tr.shape(0);
    tempShape[1] = mat2Tr.shape(1);
    AscendTensor matResultTr;
    makeTensor(ctx, matResultTr, tempShape, mat1Tr.dtype());
    diopiTensorHandle_t matResult = const_cast<diopiTensorHandle_t>(matResultTr.tensorHandle());
    diopiMm(ctx, matResult, mat1, mat2);

    // calculate beta x input;
    const std::vector<int64_t>& inputShape = inputTr.shape();
    AscendTensor bXinputTr;
    makeTensor(ctx, bXinputTr, inputShape, inputTr.dtype());
    diopiTensorHandle_t bXinput = const_cast<diopiTensorHandle_t>(bXinputTr.tensorHandle());
    diopiMulScalar(ctx, bXinput, input, beta);

    // calculate beta x input + alpha x (mat1 @ mat2);
    diopiAdd(ctx, out, bXinput, matResult, alpha);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
