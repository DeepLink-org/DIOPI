#include "../error.hpp"

#include <diopi/functions.h>

#include <cstdio>

namespace impl {
namespace kunlunxin {

char strLastError[8192] = {0};
char strLastErrorOther[4096] = {0};
std::mutex mtxLastError;

const char* klx_get_last_error_string() {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "klx error: %s", strLastErrorOther);
    return strLastError;
}

extern "C" DIOPI_RT_API const char* diopiGetLastErrorString() { return klx_get_last_error_string(); }

}  // namespace kunlunxin
}  // namespace impl
