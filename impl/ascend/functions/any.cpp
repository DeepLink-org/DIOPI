/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <limits>
#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" DIOPI_API DIOPI_API diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    long long min = std::numeric_limits<long long>::min();

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiTensorHandle_t tmpInput;
    if (diopi_dtype_bool != dtype) {
        makeTensorLike(ctx, &tmpInput, input, diopi_dtype_bool);
        diopiCastDtype(ctx, tmpInput, input);
    } else {
        tmpInput = const_cast<diopiTensorHandle_t>(input);
    }

    diopiSize_t dims;
    if (nullptr == dim || *dim == min) {
        diopiGetTensorShape(input, &dims);
        std::vector<int64_t> ivec(dims.len);
        std::iota(ivec.begin(), ivec.end(), 0);
        dims = diopiSize_t{ivec.data(), static_cast<int64_t>(ivec.size())};
        AclOpRunner<2, 1>("ReduceAny", ctx).addInput(tmpInput).addConstInput(dims).addOutput(out).setAttr("keep_dims", false).run();
    } else {
        dims = diopiSize_t{dim, 1};
        AclOpRunner<2, 1>("ReduceAny", ctx).addInput(tmpInput).addConstInput(dims).addOutput(out).setAttr("keep_dims", false).run();
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
