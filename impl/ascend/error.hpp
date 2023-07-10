#ifndef IMPL_ASCEND_ERROR_HPP_
#define IMPL_ASCEND_ERROR_HPP_

#include <mutex>
#include <utility>

namespace impl {

namespace ascend {

extern char strLastError[8192];
extern char strLastErrorOther[4096];
extern std::mutex mtxLastError;

template <typename... Types>
inline void set_last_error_string(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, szFmt, std::forward<Types>(args)...);
}

const char* ascend_get_last_error_string();

}  // namespace ascend

}  // namespace impl

#endif  // IMPL_ASCEND_ERROR_HPP_
