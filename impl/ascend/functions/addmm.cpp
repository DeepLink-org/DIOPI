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
    AscendTensor asInput(input);
    AscendTensor asMat1(mat1);
    AscendTensor asMat2(mat2);

    // calculate (mat1 @ mat2)
    std::vector<int64_t> tempShape(2);
    tempShape[0] = asMat1.shape(0);
    tempShape[1] = asMat2.shape(1);
    AscendTensor asMatResult;
    diopiDtype_t highDType = promoteTypes(asMat1.dtype(), asMat2.dtype());
    makeTensor(ctx, asMatResult, tempShape, highDType);
    diopiTensorHandle_t matResult = const_cast<diopiTensorHandle_t>(asMatResult.tensorHandle());
    diopiMm(ctx, matResult, mat1, mat2);

    // calculate beta x input;
    const std::vector<int64_t>& inputShape = asInput.shape();
    AscendTensor asBXinput;
    diopiDtype_t highDType2 = promoteTypes(asInput.dtype(), beta->stype);
    makeTensor(ctx, asBXinput, inputShape, highDType2);
    diopiTensorHandle_t bXinput = const_cast<diopiTensorHandle_t>(asBXinput.tensorHandle());
    diopiMulScalar(ctx, bXinput, input, beta);

    // calculate beta x input + alpha x (mat1 @ mat2);
    diopiAdd(ctx, out, bXinput, matResult, alpha);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
