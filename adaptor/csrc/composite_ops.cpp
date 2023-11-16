#include "impl_functions.hpp"

namespace impl {
namespace composite {

// This is an example for developing composite ops
// diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
//                          diopiConstTensorHandle_t bias) {
//     // Your logic here...
//     DIOPI_CALL(diopiMatmul(ctx, out, input, weight));
//     // Your logic here...
//     DIOPI_CALL(diopiAddInp(ctx, input, bias, {1}));
//     // Your logic here...
//     return diopiSuccess;
// }
}  // namespace composite
}  // namespace impl