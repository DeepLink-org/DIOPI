/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_IMPL_CAMB_ERROR_HPP_
#define DIOPI_IMPL_CAMB_ERROR_HPP_

#include <cnrt.h>
#include <diopi/diopirt.h>

#include <mutex>
#include <string>
#include <utility>

namespace impl {

namespace camb {

extern char strLastError[8192];
extern char strLastErrorOther[4096];
extern std::mutex mtxLastError;

template <typename... Types>
inline void setLastErrorString(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, szFmt, std::forward<Types>(args)...);
}

const char* cambGetLastErrorString();

const char* getDiopiErrorStr(diopiError_t err);

}  // namespace camb

}  // namespace impl

#endif  // DIOPI_IMPL_CAMB_ERROR_HPP_
