#include <mutex>
#include <cuda_runtime.h>

#include "error.hpp"

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

const char* cuda_get_last_error_string()
{
    cudaError_t error = cudaGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError, "cuda error: %s; other error: %s",
            cudaGetErrorString(error), strLastErrorOther);
    return strLastError;
}

void _set_last_error_string(const char *err) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, "%s", err);
}
