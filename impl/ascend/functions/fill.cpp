/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <cmath>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (numel <= 0) {
        return diopiSuccess;
    }

    float val = getValue<float>(value);

    AscendTensor inputAt(input);
    AclOpRunner<1, 1>("Fills", ctx).addInput(input).setAttr<float>("value", val).addOutput(input).run();
    diopiScalar_t zeroValueScalar = constructDiopiScalarT(inputAt.dtype(), 0.0);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
