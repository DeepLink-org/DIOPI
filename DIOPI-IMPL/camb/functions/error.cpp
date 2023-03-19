/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include "../error.hpp"
#include <diopi/functions.h>

namespace impl {
namespace camb {

char strLastError[8192] = {0};
char strLastErrorOther[4096] = {0};
std::mutex mtxLastError;

const char* camb_get_last_error_string() {
    // consider cnrt version cnrtGetLastErr or cnrtGetLaislhhstError
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "camb error: %s, more infos: %s", ::cnrtGetErrorStr(err), strLastErrorOther);
    return strLastError;
}

extern "C" DIOPI_RT_API const char* diopiGetLastErrorString() { return camb_get_last_error_string(); }

}  // namespace camb

}  // namespace impl
