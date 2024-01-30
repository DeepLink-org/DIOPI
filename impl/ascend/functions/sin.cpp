/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiSin(ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    if (useAclnn()) {
        AclTensor inAcl(input), outAcl(out);
        if (!inAcl.defined() || inAcl.numel() == 0) {
            return diopiSuccess;
        }

        aclTensor* selfAclPtr = nullptr;
        aclTensor* outAclPtr = nullptr;
        createAclTensor(input, &selfAclPtr);
        createAclTensor(out, &outAclPtr);
        ACLNN_ADAPTOR("aclnnSin", ctx, selfAclPtr, outAclPtr);
    } else {
        AscendTensor in(input);
        if (0 == in.numel()) {
            return diopiSuccess;
        }

        std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64, diopi_dtype_complex64, diopi_dtype_complex128};

        // only support: float16, float32, int32, int64, double, complex64, complex128.
        if (typeSet.find(in.dtype()) == typeSet.end()) {
            AscendTensor inputA, outA, inputTmp(input), outTmp(out);
            makeTensorLike(ctx, outA, in, diopi_dtype_float32);
            makeTensorLike(ctx, inputA, in, diopi_dtype_float32);
            castTensor(ctx, inputTmp, inputA);
            AclOpRunner<1, 1>("Sin", ctx).addInput(inputA).addOutput(outA).run();
            diopiCastDtype(ctx, out, static_cast<diopiConstTensorHandle_t>(outA));
        } else {
            AclOpRunner<1, 1>("Sin", ctx).addInput(input).addOutput(out).run();
        }
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
