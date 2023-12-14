/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "error.hpp"

#include <mutex>

extern "C" {

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

const char* cudaGetLastErrorString() {
    std::lock_guard<std::mutex> lock(mtxLastError);
    return strLastError;
}

void setLastErrorString(const char* err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}

}  // extern "C"
