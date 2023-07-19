#include <diopi/functions.h>

#include <iostream>

#include <memory>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {


extern "C" DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                                         diopiSize_t output_size) {
    AclOpRunner<1, 1> runner("AdaptiveAvgPool2d");
    runner.addInput(input);
    runner.setAttr("output_size", std::vector<int32_t>{output_size.data[0], output_size.data[1]});
    runner.addOutput(out);
    runner.run(ctx);
    return diopiSuccess;
}


extern "C" DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                                 diopiConstTensorHandle_t input) {
    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    AclOpRunner<1, 1> runner("AdaptiveAvgPool2dGrad");
    runner.addInput(grad_output);
    runner.setAttr("orig_input_shape", std::vector<int32_t>{shape.data[0], shape.data[1], shape.data[2], shape.data[3]});
    runner.addOutput(grad_input);
    runner.run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
