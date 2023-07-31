/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                                         diopiSize_t output_size) {
    AclOpRunner<1, 1>("AdaptiveAvgPool2d")
        .addInput(input)
        .setAttr("output_size", std::vector<int32_t>{output_size.data[0], output_size.data[1]})
        .addOutput(out)
        .run(ctx);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                                 diopiConstTensorHandle_t input) {
    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    AclOpRunner<1, 1>("AdaptiveAvgPool2dGrad")
        .addInput(grad_output)
        .setAttr("orig_input_shape", std::vector<int32_t>{shape.data[0], shape.data[1], shape.data[2], shape.data[3]})
        .addOutput(grad_input)
        .run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
