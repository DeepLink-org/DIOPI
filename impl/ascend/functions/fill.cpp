#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    AclOpRunner<1, 1>("Fills").addInput(input).setAttr<float>("value", getValue<float>(value)).addOutput(input).run(ctx);
    return diopiSuccess;
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
