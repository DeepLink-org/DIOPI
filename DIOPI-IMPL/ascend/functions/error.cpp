#include "../error.hpp"

#include <acl/acl.h>
#include <diopi/functions.h>
#include <cstdio>

namespace impl {
namespace ascend {

char strLastError[8192] = {0};
char strLastErrorOther[4096] = {0};
std::mutex mtxLastError;

const char* ascend_get_last_error_string() {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "ascend error: %s, more infos: %s", ::aclGetRecentErrMsg(), strLastErrorOther);
    return strLastError;
}

extern "C" DIOPI_RT_API const char* diopiGetLastErrorString() { return ascend_get_last_error_string(); }

}  // namespace ascend

}  // namespace impl
