#include <diopi/functions.h>

#include <algorithm>
#include <numeric>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    auto out_tensor = DiopiTensor(out);
    long int arr[n];
    std::iota(arr, arr + n, 0);
    std::random_shuffle(arr, arr + n);
    auto ret = cnrtMemcpy(out_tensor.data(), arr, sizeof(long int) * n, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        set_last_error_string("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
