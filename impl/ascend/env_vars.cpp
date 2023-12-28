/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include "env_vars.hpp"

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

bool isDebugAclOpRunnerOn() {
    static bool isDebugAclOpRunner = isEnvStateOn("DIOPI_DEBUG_ACLOPRUNNER");
    return isDebugAclOpRunner;
}
