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
    std::cout << "ready to run diopiAddmm" << std::endl;
    AscendTensor asInput(input);
    AscendTensor asOut(out);
    std::cout << "FLAG1" << std::endl;
    // calculate (mat1 @ mat2)
    std::vector<int64_t> outShape = asOut.shape();
    diopiSize_t outShape_ = vectorToDiopiSize(outShape);
    diopiTensorHandle_t matResult;
    makeTensorFromSize(ctx, &outShape_, &matResult, asOut.dtype());
    diopiMm(ctx, matResult, mat1, mat2);

    std::cout << "FLAG2" << std::endl;
    // calculate beta x input;
    std::vector<int64_t> inputShape = asInput.shape();
    diopiSize_t inputShape_ = vectorToDiopiSize(inputShape);
    diopiTensorHandle_t bXinput;
    makeTensorFromSize(ctx, &inputShape_, &bXinput, asOut.dtype());
    diopiMulScalar(ctx, bXinput, input, beta);

    std::cout << "FLAG3" << std::endl;
    // calculate beta x input + alpha x (mat1 @ mat2);
    diopiAdd(ctx, out, bXinput, matResult, alpha);
}

}  // namespace ascend
}  // namespace impl
