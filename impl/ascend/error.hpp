/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_ERROR_HPP_
#define IMPL_ASCEND_ERROR_HPP_

#include <diopi/diopirt.h>

#include <cstring>
#include <mutex>
#include <utility>

namespace impl {

namespace ascend {

extern char strLastError[8192];
extern int32_t curIdxError;
extern std::mutex mtxLastError;

template <typename... Types>
inline void setLastErrorString(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError + curIdxError, szFmt, std::forward<Types>(args)...);
    curIdxError = strlen(strLastError);
}

const char* ascendGetLastErrorString(bool clearBuff);
const char* getDiopiErrorStr(diopiError_t err);
}  // namespace ascend

}  // namespace impl

#endif  // IMPL_ASCEND_ERROR_HPP_
