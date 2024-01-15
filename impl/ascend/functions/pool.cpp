/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"
namespace impl {
namespace ascend {

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t outputSize) {
    if (0 == outputSize.data[0]) {
        return diopiSuccess;
    }

    AclOpRunner<1, 1>("AdaptiveAvgPool2d", ctx)
        .addInput(input)
        .setAttr("output_size", std::vector<int32_t>{static_cast<int>(outputSize.data[0]), static_cast<int>(outputSize.data[1])})
        .addOutput(out)
        .run();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                            diopiConstTensorHandle_t input) {
    diopiSize_t shape;
    diopiGetTensorShape(input, &shape);
    diopiSize_t gradShape;
    diopiGetTensorShape(gradOutput, &gradShape);
    if (gradShape.data[gradShape.len - 1] == gradShape.data[gradShape.len - 2] && 1 == gradShape.data[gradShape.len - 1]) {
        float temp = shape.data[shape.len - 1] * shape.data[shape.len - 2];
        temp = temp == 0 ? 1 : temp;
        temp = 1 / temp;
        diopiScalar_t scalarTemp = constructDiopiScalarT(diopi_dtype_float64, temp);
        diopiFill(ctx, gradInput, &scalarTemp);
        diopiMulInp(ctx, gradInput, gradOutput);
        return diopiSuccess;
    }
    std::vector<int32_t> shapeVector;
    shapeVector.reserve(shape.len);
    for (int i = 0; i < shape.len; ++i) {
        shapeVector.push_back(static_cast<int>(shape.data[i]));
    }
    AclOpRunner<1, 1>("AdaptiveAvgPool2dGrad", ctx).addInput(gradOutput).setAttr("orig_input_shape", shapeVector).addOutput(gradInput).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
