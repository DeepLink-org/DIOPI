
#include <mutex>
#include <cnrt.h>

#include "error.hpp"

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

const char* camb_get_last_error_string() {
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "camb error: %s; other error: %s",
            ::cnrtGetErrorStr(err), strLastErrorOther);
    return strLastError;
}

void _set_last_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}
