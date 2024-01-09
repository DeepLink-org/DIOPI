/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t scatterTensorInp(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t dim, diopiConstTensorHandle_t src, const AscendTensor& indexAt,
                              const char* reduce) {
    std::string reduction;
    if (strcmp(reduce, "") == 0) {
        reduction = "none";
    } else if (strcmp(reduce, "add") == 0) {
        reduction = "add";
    } else if (strcmp(reduce, "multiply") == 0) {
        reduction = "mul";
    }
    AclOpRunner<3, 1>("ScatterElements", ctx)
        .addInput(out)
        .addInput(indexAt)
        .addInput(src)
        .addOutput(out)
        .setAttr("axis", dim)
        .setAttr("reduction", reduction)
        .run();
    return diopiSuccess;
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                          diopiConstTensorHandle_t index, const char* reduce) {
    diopiCastDtype(ctx, out, input);
    return scatterTensorInp(ctx, out, dim, src, AscendTensor(index), reduce);
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    return scatterTensorInp(ctx, input, dim, src, AscendTensor(index), reduce);
}

void prepareScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* value, diopiTensorHandle_t& src, AscendTensor& indexAt) {
    // Shape of valueTh: {1,}
    diopiTensorHandle_t valueTh;
    makeTensorFromScalar(ctx, value, &valueTh, diopi_device);

    // Ensure that src has the same dtype as out and the same shape as index
    if (indexAt.shape().empty()) {
        makeTensorLike(ctx, &src, valueTh, impl::ascend::AscendTensor(out).dtype());
        diopiCastDtype(ctx, src, valueTh);
        indexAt.view({1});
    } else {
        makeTensorLike(ctx, &src, indexAt.tensorHandle(), impl::ascend::AscendTensor(out).dtype());
        broadcast(ctx, src, valueTh, indexAt.shape());
    }
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    diopiCastDtype(ctx, out, input);

    diopiTensorHandle_t src;
    AscendTensor indexAt(index);
    prepareScatterScalar(ctx, out, value, src, indexAt);
    return scatterTensorInp(ctx, out, dim, src, indexAt, reduce);
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    diopiTensorHandle_t src;
    AscendTensor indexAt(index);
    prepareScatterScalar(ctx, input, value, src, indexAt);
    return scatterTensorInp(ctx, input, dim, src, indexAt, reduce);
}

}  // namespace ascend
}  // namespace impl
