/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    AscendTensor AsCondition(condition);
    AscendTensor AsInput(input);
    AscendTensor AsOther(other);
    
    if (AsCondition.dtype() != diopi_dtype_bool) {
        castTensor(ctx, AsCondition, diopi_dtype_bool);
    }


    AclOpRunner<3, 1>("Select", ctx).addInput(AsCondition).addInput(input).addInput(other).addOutput(out).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
