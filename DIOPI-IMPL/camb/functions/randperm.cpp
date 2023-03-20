#include <diopi/functions.h>

#include <algorithm>
#include <numeric>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

namespace {
template <typename T>
diopiError_t randperm_func(DiopiTensor tensor, int64_t n, int64_t idx) {
    T array[n];
    std::iota(array, array + n, 0);
    std::random_shuffle(array, array + n);
    auto ret = cnrtMemcpy(tensor.data(), array, sizeof(T) * n, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        set_last_error_string("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}
}  // namespace

extern "C" DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    auto out_tensor = DiopiTensor(out);
    if (out_tensor.dtype() == diopi_dtype_int32) {
        DIOPI_CALL(randperm_func<int>(out_tensor, n, idx));
    } else if (out_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(randperm_func<long int>(out_tensor, n, idx));
    } else {
        set_last_error_string("randperm not support datatype %d.\n", out_tensor.dtype());
        return diopi5DNotSupported;
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
