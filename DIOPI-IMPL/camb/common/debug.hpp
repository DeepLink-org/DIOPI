/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#ifndef DIOPI_IMPL_CAMB_COMMON_DEBUG_HPP_
#define DIOPI_IMPL_CAMB_COMMON_DEBUG_HPP_

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <memory>

#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

// print the data on dev
// T is the dtype such as  float, int64_t, double.
template <typename T>
void printDevData(diopiContextHandle_t ctx, void* data, int64_t len, int64_t max_len, T) {
    int bytes = sizeof(T) * len;
    std::unique_ptr<char> ptr(new char[bytes]);
    std::cout << "data address:" << data << std::endl;
    cnrtMemcpyAsync(ptr.get(), data, bytes, getStream(ctx), cnrtMemcpyDevToHost);
    syncStreamInCtx(ctx);
    for (int i = 0; i < len && i < max_len; ++i) {
        std::cout << reinterpret_cast<T*>(ptr.get())[i] << " ";
    }
    std::cout << std::endl;
}

template<>
void printDevData<int8_t>(diopiContextHandle_t ctx, void* data, int64_t len, int64_t max_len, int8_t _) {
    int bytes = len;
    std::unique_ptr<char> ptr(new char[bytes]);
    std::cout << "data address:" << data << std::endl;
    cnrtMemcpyAsync(ptr.get(), data, bytes, getStream(ctx), cnrtMemcpyDevToHost);
    syncStreamInCtx(ctx);
    for (int i = 0; i < len && i < max_len; ++i) {
        std::cout << static_cast<int32>(reinterpret_cast<T*>(ptr.get())[i]) << " ";
    }
    std::cout << std::endl;
}

template<>
void printDevData<uint8_t>(diopiContextHandle_t ctx, void* data, int64_t len, int64_t max_len, uint8_t _) {
    printDevData<int8_t>(ctx, data, len, max_len, static_cast<int8_t>(_));
}

static void print_backtrace() {
    const int MAX_STACK_FRAMES = 64;
    void* stack_traces[MAX_STACK_FRAMES];
    int stack_frames = backtrace(stack_traces, MAX_STACK_FRAMES);

    char** stack_strings = backtrace_symbols(stack_traces, stack_frames);  // do not forget to free stack_strings
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

#endif  // DIOPI_IMPL_CAMB_COMMON_DEBUG_HPP_
