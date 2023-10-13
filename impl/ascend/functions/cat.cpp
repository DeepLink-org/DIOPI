/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    
    std::vector<diopiConstTensorHandle_t> dynamicInput;
    for (int i = 0; i < numInputs; i++) {
        int64_t tem;
        diopiGetTensorNumel(tensors[i], &tem);
        if (tem != 0) dynamicInput.push_back(tensors[i]);
    }
    diopiDtype_t outDtype, castType;
    diopiGetTensorDtype(out, &outDtype);

    if (isFloatingType(outDtype))
        castType = diopi_dtype_float32;
    else
        castType = diopi_dtype_int32;

    AscendTensor outA(out);
    castTensor(ctx, outA, castType);

    numInputs = dynamicInput.size();
    if (numInputs == 1) {
        diopiCastDtype(ctx, out, const_cast<diopiTensorHandle_t>(dynamicInput[0]));
        return diopiSuccess;
    }
    AclOpRunner<1, 1>("ConcatD", ctx).addDynamicInput(dynamicInput, castType).setAttr("N", numInputs).setAttr("concat_dim", dim).addOutput(outA).run();
    castTensor(ctx, outA, outDtype);
    diopiCastDtype(ctx, out, static_cast<diopiConstTensorHandle_t>(outA));
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
