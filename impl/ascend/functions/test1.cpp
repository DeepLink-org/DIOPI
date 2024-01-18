/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>
#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

extern "C" {
DIOPI_API diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    check_args(numInputs <= 2048, "cannot concat more than 2048 tensors");
    if (numInputs <= 2) {
        AclOpRunner<2, 1> runner("ConcatD", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("concat_dim", dim).addOutput(out).run();
    } else if (numInputs <= 8) {
        ...
    } else if (numInputs <= 2048) {
        AclOpRunner<2048, 1> runner("ConcatD", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("concat_dim", dim).addOutput(out).run();
    }
    return diopiSuccess;
}
}