/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"
using namespace std;

namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    AclOpRunner<1, 1>("Sqrt", ctx).addInput(input).addOutput(out).run();

    // 解决ascend对负数做sqrt不返回nan的问题
    // get zero value tensor
    diopiTensorHandle_t zeroValue;
    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &zeroValueScalar, &zeroValue, diopi_dtype_float32, diopi_device);
    // get nan value tensor
    diopiTensorHandle_t nanValue;
    auto nanValueScalar = diopiScalar_t();
    nanValueScalar.stype = diopi_dtype_float32;
    nanValueScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &nanValueScalar, &nanValue, diopi_dtype_float32, diopi_device);
    diopiDivInpScalar(ctx, nanValue, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);
    // get nagetive mask
    diopiTensorHandle_t mask;
    makeTensorLike(ctx, &mask, input, diopi_dtype_bool);
    diopiLtScalar(ctx, mask, input, &zeroValueScalar);
    diopiMaskedFillInp(ctx, out, mask, nanValue);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
