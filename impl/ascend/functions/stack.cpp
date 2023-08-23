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
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numInputs, int64_t dim) {
    check_args(numInputs <= 2048, "cannot stack more than 2048 tensors");
    if (numInputs <= 2) {
        AclOpRunner<2, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    } else if (numInputs <= 8) {
        AclOpRunner<8, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    } else if (numInputs <= 32) {
        AclOpRunner<32, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    } else if (numInputs <= 128) {
        AclOpRunner<128, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    } else if (numInputs <= 512) {
        AclOpRunner<512, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    } else if (numInputs <= 2048) {
        AclOpRunner<2048, 1> runner("Pack", ctx);
        for (int64_t i = 0; i < numInputs; i++) {
            runner.addInput(tensors[i]);
        }
        runner.setAttr("N", numInputs).setAttr("axis", dim).addOutput(out).run();
    }
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
