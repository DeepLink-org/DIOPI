#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias) {
    AclOpRunner<3, 1> runner("MatMul");
    runner.addInput(input, weight).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true).addOutput(out);
    if (bias) runner.addInput(bias);
    runner.run(ctx);
    return diopiSuccess;
}

extern "C" diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                            diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight) {
    AclOpRunner<2, 1>("MatMul")
        .addInput(gradOutput, weight)
        .setAttr<uint8_t>("transpose_x1", false)
        .setAttr<uint8_t>("transpose_x2", false)
        .addOutput(gradInput)
        .run(ctx);

    AclOpRunner<2, 1>("MatMul")
        .addInput(gradOutput, input)
        .setAttr<uint8_t>("transpose_x1", true)
        .setAttr<uint8_t>("transpose_x2", false)
        .addOutput(gradWeight)
        .run(ctx);

    if (gradBias) {
        AclOpRunner<1, 1>("Fills").addInput(gradBias).setAttr<float>("value", 1.0f).addOutput(gradBias).run(ctx);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
