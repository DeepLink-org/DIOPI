/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace impl {
namespace camb {

void print_backtrace() {
    const int MAX_STACK_FRAMES = 64;
    void* stack_traces[MAX_STACK_FRAMES];
    int stack_frames = backtrace(stack_traces, MAX_STACK_FRAMES);

    char** stack_strings = backtrace_symbols(stack_traces, stack_frames);
    for (int i = 0; i < stack_frames; i++) {
        printf("%s\n", stack_strings[i]);

        // Try to demangle the symbol name
        char* symbol = stack_strings[i];
        char* mangled_start = strchr(symbol, '(');
        char* mangled_end = strchr(mangled_start, '+');
        if (mangled_start && mangled_end) {
            *mangled_start++ = '\0';
            *mangled_end = '\0';
            int status;
            char* demangled = abi::__cxa_demangle(mangled_start, nullptr, nullptr, &status);
            if (status == 0) {
                printf("  %s\n", demangled);
            }
        }
    }
    free(stack_strings);
}

}  // namespace camb
}  // namespace impl
