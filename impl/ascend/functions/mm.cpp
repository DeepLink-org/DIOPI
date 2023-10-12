/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

<<<<<<< HEAD
extern "C" diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    //AclOpRunner<2, 1>("BatchMatMul", ctx).addInput(input).addInput(mat2).setAttr("adj_x1", true).setAttr("adj_x1", true).addOutput(out).run();
    AscendTensor inputCopy(input);
    AscendTensor mat2Copy(mat2);
    diopiDtype_t highDType = promoteTypes(inputCopy.dtype(), mat2Copy.dtype());
    if(highDType == diopi_dtype_float64)
        highDType = diopi_dtype_float32;
    printContiguousTensor(ctx, inputCopy, "input");
    printContiguousTensor(ctx, mat2Copy, "mat2");
    AclOpRunner<2, 1>("MatMul", ctx).addInput(input, highDType).addInput(mat2, highDType).setAttr("adj_x1", false).setAttr("adj_x1", false).addOutput(out).run();
    AscendTensor outputCopy(out);
    printContiguousTensor(ctx, outputCopy, "output");
=======
diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AclOpRunner<2, 1>("BatchMatMul", ctx).addInput(input).addInput(mat2).setAttr("adj_x1", true).setAttr("adj_x1", true).addOutput(out).run();
>>>>>>> 0f88709f89431550684a045e0306a2e26ce6abb7
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
