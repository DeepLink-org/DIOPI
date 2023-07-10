/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CUDA_ERROR_HPP_
#define IMPL_CUDA_ERROR_HPP_

#include <utility>

extern "C" {

void _set_last_error_string(const char* err);
const char* cuda_get_last_error_string();

}  // extern "C"

namespace impl {

namespace cuda {

template <typename... Types>
void set_last_error_string(const char* szFmt, Types&&... args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

}  // namespace cuda

}  // namespace impl

#endif  // IMPL_CUDA_ERROR_HPP_
