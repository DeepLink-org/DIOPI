/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../error.hpp"

#include <acl/acl.h>
#include <diopi/functions.h>

#include <cstdio>

namespace impl {
namespace ascend {

char strLastError[8192] = {0};
char strLastErrorOther[4096] = {0};
std::mutex mtxLastError;

const char* ascendGetLastErrorString() {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "ascend error: %s, more infos: %s", ::aclGetRecentErrMsg(), strLastErrorOther);
    return strLastError;
}

extern "C" DIOPI_RT_API const char* diopiGetLastErrorString() { return ascendGetLastErrorString(); }

}  // namespace ascend

}  // namespace impl
