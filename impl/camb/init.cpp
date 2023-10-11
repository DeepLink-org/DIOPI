/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cstdlib>
#include <cstring>

bool isEnvStateOn(const char* envName) {
    char* val = getenv(envName);
    // the val is nullptr, 0, OFF or off will not turn the record on, otherwise turn it off
    if (!val || !strcmp(val, "0") || !strcmp(val, "OFF") || !strcmp(val, "off")) {
        // turn record on
        return true;
    }
    return false;
}

bool isRecordOnFunc() { return isEnvStateOn("DIOPI_RECORD_ENV"); }

// global vars are listed bellow
bool isRecordOn;

int initFunc() {
    // init func is called  here;
    isRecordOn = isRecordOnFunc();
    return 0;
}

static int initGlobalVal = initFunc();
