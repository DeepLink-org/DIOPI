/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    AscendTensor inputCopy(input);
    AscendTensor mat2Copy(mat2);
    diopiDtype_t dtype = inputCopy.dtype();
    if(dtype == diopi_dtype_float64)
        dtype = diopi_dtype_float32;
    printContiguousTensor(ctx, inputCopy, "input");
    printContiguousTensor(ctx, mat2Copy, "mat2");
    AclOpRunner<2, 1>("BatchMatMul", ctx).addInput(input).addInput(mat2).setAttr("adj_x1", false).setAttr("adj_x1", false).addOutput(out).run();
    AscendTensor outputCopy(out);
    printContiguousTensor(ctx, outputCopy, "output");
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
