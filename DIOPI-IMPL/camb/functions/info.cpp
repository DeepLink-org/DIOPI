/**************************************************************************************************
* @Copyright (c) 2023, SenseTime Inc.
*
*************************************************************************************************/

#include <cnnl.h>
#include <cnrt.h>
#include <diopi/functions.h>

#include <cstdio>
#include <cstring>

#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

static char version[512];

extern "C" DIOPI_RT_API const char* diopiGetVendorName() { return "CambDevice"; }
extern "C" DIOPI_RT_API const char* diopiGetImplVersion() {
    if (strlen(version) == 0) {
        sprintf(version, "Cnrt Version: %d; CNNL Version: %d; DIOPI Version: %d", CNRT_VERSION, CNNL_VERSION, DIOPI_VERSION);
    }
    return version;
}

}  // namespace camb
}  // namespace impl
