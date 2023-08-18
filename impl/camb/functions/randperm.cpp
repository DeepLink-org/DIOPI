/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <algorithm>
#include <numeric>
#include <random>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

namespace {
template <typename T>
diopiError_t randpermFunc(DiopiTensor tensor, int64_t n, int64_t idx) {
    std::vector<T> vec(n);
    std::iota(vec.begin(), vec.end(), 0);
    std::shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));
    auto ret = cnrtMemcpy(tensor.data(), vec.data(), sizeof(T) * n, cnrtMemcpyHostToDev);
    if (ret != cnrtSuccess) {
        setLastErrorString("%s%d", "cnrt memcpy error, ret = ", ret);
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n) {
    DiopiTensor outTensor(out);
    if (outTensor.dtype() == diopi_dtype_int32) {
        DIOPI_CALL(randpermFunc<int>(outTensor, n, idx));
    } else if (outTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(randpermFunc<long int>(outTensor, n, idx));
    } else {
        setLastErrorString("randperm not support datatype %d.\n", outTensor.dtype());
        return diopi5DNotSupported;
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
