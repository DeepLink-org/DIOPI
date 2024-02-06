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
        int64_t numel;
        diopiGetTensorNumel(tensors[i], &numel);
        if (numel != 0) dynamicInput.push_back(tensors[i]);
    }

    AscendTensor outAt(out);
    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);

    numInputs = dynamicInput.size();
    AclOpRunner<1, 1>("ConcatD", ctx).addDynamicInput(dynamicInput, outDtype).setAttr("N", numInputs).setAttr("concat_dim", dim).addOutput(outAt).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
