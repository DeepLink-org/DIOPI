/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_ERROR_HPP_
#define IMPL_CAMB_ERROR_HPP_

#include <cnrt.h>
#include <cxxabi.h>
#include <diopi/diopirt.h>
#include <execinfo.h>

#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

namespace impl {

namespace camb {

extern char strLastError[8192];
extern int32_t curIdxError;
extern std::mutex mtxLastError;

template <typename... Types>
inline void setLastErrorString(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastError + curIdxError, szFmt, std::forward<Types>(args)...);
    curIdxError = strlen(strLastError);
}

const char* cambGetLastErrorString(bool clearBuff);

const char* getDiopiErrorStr(diopiError_t err);

#ifdef DEBUG_MODE
inline void printBacktrace() {
    const int maxStackFrames = 64;
    void* stackTraces[maxStackFrames];
    int stackFrames = backtrace(stackTraces, maxStackFrames);

    char** stackStrings = backtrace_symbols(stackTraces, stackFrames);  // do not forget to free stack_strings
    for (int i = 0; i < stackFrames; i++) {
        printf("%s\n", stackStrings[i]);

        // Try to demangle the symbol name
        char* symbol = stackStrings[i];
        char* mangledStart = strchr(symbol, '(');
        char* mangledEnd = strchr(mangledStart, '+');
        if (mangledStart && mangledEnd) {
            *mangledStart++ = '\0';
            *mangledEnd = '\0';
            int status;
            char* demangled = abi::__cxa_demangle(mangledStart, nullptr, nullptr, &status);
            if (status == 0) {
                printf("  %s\n", demangled);
            }
        }
    }
    free(stackStrings);
}
#endif

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_ERROR_HPP_
