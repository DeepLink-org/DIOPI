/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inputCopy(input);
    AscendTensor mat2Copy(mat2);
    AscendTensor outputCopy(out);

    if (inputCopy.numel() == 0 || mat2Copy.numel() == 0) {
        std::cout << "fill 0" << std::endl;
        diopiScalar_t zero = {outputCopy.dtype(), 0.0};
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }

    auto inputShape = inputCopy.shape();
    std::cout << "input shape:" << std::endl;
    for (auto i : inputShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    auto mat2Shape = mat2Copy.shape();
    std::cout << "mat2 shape:" << std::endl;
    for (auto i : mat2Shape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    diopiDtype_t highDType = promoteTypes(inputCopy.dtype(), mat2Copy.dtype());
    if (highDType == diopi_dtype_float64) highDType = diopi_dtype_float32;
    AclOpRunner<2, 1>("MatMul", ctx)
        .addInput(input, highDType)
        .addInput(mat2, highDType)
        .setAttr("adj_x1", false)
        .setAttr("adj_x1", false)
        .addOutput(out)
        .run();
    AscendTensor outputCopyV2(out);

    auto outputShape = outputCopyV2.shape();
    std::cout << "output shape:" << std::endl;
    for (auto i : outputShape) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
