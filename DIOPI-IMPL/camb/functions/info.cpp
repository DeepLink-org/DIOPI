#include <cnnl.h>
#include <cnrt.h>
#include <diopi/functions.h>

#include <cstdio>
#include <cstring>

#include "../diopi_helper.hpp"

static char version[512];

const char* diopiGetVendorName() { return "CambDevice"; }
const char* diopiGetImplVersion() {
    if (strlen(version) == 0) {
        sprintf(version, "Cnrt Version: %d; CNNL Version: %d; DIOPI Version: %d", CNRT_VERSION, CNNL_VERSION, DIOPI_VERSION);
    }
    return version;
}
