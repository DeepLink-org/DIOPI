#ifndef IMPL_CAMB_ERROR_HPP_
#define IMPL_CAMB_ERROR_HPP_

#include <cnrt.h>

#include <mutex>
#include <utility>

namespace impl {

namespace camb {

extern char strLastError[8192];
extern char strLastErrorOther[4096];
extern std::mutex mtxLastError;

template <typename... Types>
inline void set_last_error_string(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, szFmt, std::forward<Types>(args)...);
}

const char* camb_get_last_error_string();

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_ERROR_HPP_
