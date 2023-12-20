#ifndef IMPL_KUNLUNXIN_ERROR_HPP_
#define IMPL_KUNLUNXIN_ERROR_HPP_

#include <mutex>
#include <utility>

namespace impl {
namespace kunlunxin {

extern char strLastError[8192];
extern char strLastErrorOther[4096];
extern std::mutex mtxLastError;

template <typename... Types>
inline void set_last_error_string(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, szFmt, std::forward<Types>(args)...);
}

const char* klx_get_last_error_string();

}  // namespace kunlunxin
}  // namespace impl

#endif  // IMPL_KUNLUNXIN_ERROR_HPP_
