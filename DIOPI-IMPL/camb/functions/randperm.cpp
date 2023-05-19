/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <algorithm>
#include <numeric>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

namespace {
template <typename T>
diopiError_t randpermFunc(DiopiTensor tensor, int64_t n, int64_t idx) {
    std::vector<T> vec(n);
    std::iota(vec.begin(), vec.end(), 0);
    std::random_shuffle(vec.begin(), vec.end());
    auto ret = cnrtMemcpy(tensor.data(), vec.data(), sizeof(T) * n, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        set_last_error_string("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    DiopiTensor outTensor(out);
    if (outTensor.dtype() == diopi_dtype_int32) {
        DIOPI_CALL(randpermFunc<int>(outTensor, n, idx));
    } else if (outTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(randpermFunc<long int>(outTensor, n, idx));
    } else {
        set_last_error_string("randperm not support datatype %d.\n", outTensor.dtype());
        return diopi5DNotSupported;
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
