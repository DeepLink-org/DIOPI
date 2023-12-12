/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <iostream>
#include <utility>

#ifndef IMPL_TORCH_ERROR_HPP_
#define IMPL_TORCH_ERROR_HPP_

void _set_last_error_string(const char* err);
const char* cuda_get_last_error_string();

inline void logError() { std::cerr << std::endl; }

template <typename First, typename... Rest>
void logError(First&& first, Rest&&... rest) {
    std::cerr << std::forward<First>(first);
    logError(std::forward<Rest>(rest)...);
}

template <typename... Types>
inline void set_last_error_string(const char* szFmt, Types&&... args) {
    char szBuf[4096] = {0};
    sprintf(szBuf, szFmt, std::forward<Types>(args)...);
    _set_last_error_string(szBuf);
}

#endif  // IMPL_TORCH_ERROR_HPP_
