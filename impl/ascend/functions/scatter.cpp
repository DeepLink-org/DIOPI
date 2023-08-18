/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" {
/*DIOPI_API diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                    diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    AclOpRunner<3, 1> runner("ScatterElements", ctx);
    runner.addInput(input, index, src).setAttr("axis", dim).addOutput(out);
    if (strlen(reduce) > 0) runner.setAttr("reduction", std::string(reduce));
    runner.run();
    info("----------reduction: ", std::string(reduce));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                                       diopiConstTensorHandle_t index, const char* reduce) {
    return diopiScatter(ctx, input, input, dim, src, index, reduce);
}

DIOPI_API diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                          const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiTensorHandle_t srcTensorBroadcast;
    makeTensorLike(ctx, &srcTensorBroadcast, index, diopi_dtype_float32);
    diopiFill(ctx, srcTensorBroadcast, value);
    return diopiScatter(ctx, out, input, dim, srcTensorBroadcast, index, reduce);
}

DIOPI_API diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                             diopiConstTensorHandle_t index, const char* reduce) {
    return diopiScatterScalar(ctx, input, input, dim, value, index, reduce);
}*/
}

}  // namespace ascend
}  // namespace impl
