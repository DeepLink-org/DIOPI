/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"
#include "../common/print.hpp"
#include <iostream>
using namespace std;

namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    printf("come info diopi sqrt");
    AclOpRunner<1, 1>("Sqrt", ctx).addInput(input).addOutput(out).run();

    // 解决ascend对负数做sqrt不返回nan的问题
    // diopiScalar_t zero{diopi_dtype_float32, {0.0f}};
    diopiTensorHandle_t zeroValue;
    auto zeroValueScalar = diopiScalar_t();
    zeroValueScalar.stype = diopi_dtype_float64;
    zeroValueScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &zeroValueScalar, &zeroValue, diopi_dtype_float32, diopi_device);
    printTensor(ctx, zeroValue, "zeroValue");
    // diopiConstTensorHandle_t zero_value{diopi_dtype_float32, {0.0f}};
    // diopiConstTensorHandle_t nan_value{diopi_dtype_float32, {std::numeric_limits<float>::quiet_NaN()}};
    diopiTensorHandle_t nanValue;
    auto nanValueScalar = diopiScalar_t();
    nanValueScalar.stype = diopi_dtype_float32;
    nanValueScalar.fval = 0.0;
    makeTensorFromScalar(ctx, &nanValueScalar, &nanValue, diopi_dtype_float32, diopi_device);
    diopiDivInpScalar(ctx, nanValue, &zeroValueScalar, diopiRoundMode_t::RoundModeNone);
    diopiTensorHandle_t mask;
    makeTensorLike(ctx, &mask, input, diopi_dtype_bool);
    diopiLtScalar(ctx, mask, input, &zeroValueScalar);
    diopiMaskedFillInp(ctx, out, mask, nanValue);
    
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl