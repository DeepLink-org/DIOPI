#include <diopi/functions.h>


#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiTensorHandle_t inputShapeT = nullptr;
    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);
    makeTensorFromSize(ctx, &inputShape, &inputShapeT);
    AclOpRunner<2, 1>("ReduceSum")
        .addInput(input, inputShapeT)
        .setAttr<uint8_t>("keep_dims", false)
        .addOutput(out)
        .run(ctx);
    return diopiSuccess;
}


extern "C" diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiTensorHandle_t inputShapeT = nullptr;
    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);
    makeTensorFromSize(ctx, &inputShape, &inputShapeT);
    AclOpRunner<2, 1>("ReduceMean")
        .addInput(input, inputShapeT)
        .setAttr<uint8_t>("keep_dims", false)
        .addOutput(out)
        .run(ctx);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
