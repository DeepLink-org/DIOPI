/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cstdlib>
#include <cstring>

bool isEnvStateOn(const char* envName) {
    char* val = getenv(envName);
    // the val is nullptr, 0, OFF or off will turn the state off, otherwise turn it on.
    if (!val || !strcmp(val, "0") || !strcmp(val, "OFF") || !strcmp(val, "off")) {
        return false;
    }
    return true;
}

bool isRecordOn() {
    static bool isRecord = isEnvStateOn("DIOPI_RECORD_ENV");
    return isRecord;
}
