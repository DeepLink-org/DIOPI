/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../error.hpp"

#include <diopi/functions.h>

namespace impl {
namespace camb {

char strLastError[8192] = {0};
int32_t curIdxError = 0;
std::mutex mtxLastError;

const char* cambGetLastErrorString() {
    // consider cnrt version cnrtGetLastErr or cnrtGetLaislhhstError
    ::cnrtRet_t err = ::cnrtGetLastError();
    std::lock_guard<std::mutex> lock(mtxLastError);
    if (cnrtSuccess != err) {
        sprintf(strLastError + curIdxError, "camb error: %s, more infos: %s", ::cnrtGetErrorStr(err), strLastError);
    }
    curIdxError = 0;
    return strLastError;
}

const char* getDiopiErrorStr(diopiError_t err) {
    switch (err) {
        case diopiErrorOccurred:
            return "diopiErrorOccurred";
        case diopiNotInited:
            return "diopiNotInited";
        case diopiNoRegisteredStreamCreateFunction:
            return "diopiNoRegisteredStreamCreateFunction";
        case diopiNoRegisteredStreamDestoryFunction:
            return "diopiNoRegisteredStreamDestoryFunction";
        case diopiNoRegisteredStreamSyncFunction:
            return "diopiNoRegisteredStreamSyncFunction";
        case diopiNoRegisteredDeviceMemoryMallocFunction:
            return "diopiNoRegisteredDeviceMemoryMallocFunction";
        case diopiNoRegisteredDeviceMemoryFreeFunction:
            return "diopiNoRegisteredDeviceMemoryFreeFunction";
        case diopiNoRegisteredDevice2DdeviceMemoryCopyFunction:
            return "diopiNoRegisteredDevice2DdeviceMemoryCopyFunction";
        case diopiNoRegisteredDevice2HostMemoryCopyFunction:
            return "diopiNoRegisteredDevice2HostMemoryCopyFunction";
        case diopiNoRegisteredHost2DeviceMemoryCopyFunction:
            return "diopiNoRegisteredHost2DeviceMemoryCopyFunction";
        case diopiNoRegisteredGetLastErrorFunction:
            return "diopiNoRegisteredGetLastErrorFunction";
        case diopi5DNotSupported:
            return "diopi5DNotSupported";
        case diopiDtypeNotSupported:
            return "diopiDtypeNotSupported";
        default:
            return "diopiUnexpectedError";
    }
}

}  // namespace camb

}  // namespace impl

const char* diopiGetLastErrorString() { return impl::camb::cambGetLastErrorString(); }
