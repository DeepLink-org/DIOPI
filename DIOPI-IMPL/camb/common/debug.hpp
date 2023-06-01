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
#include <vector>
#include <iostream>
#include <memory>

#include "../diopi_helper.hpp"

namespace impl {
namespace camb {

namespace {

// print the data on dev
// T is the dtype such as  float, int64_t, double.
template <typename RealT, typename CastT>
void printDevDataInternal(diopiContextHandle_t ctx, void* data, int64_t len, int64_t max_len) {
    int bytes = sizeof(RealT) * len;
    std::unique_ptr<char> ptr(new char[bytes]);
    std::cout << "data address:" << data << std::endl;
    cnrtMemcpyAsync(ptr.get(), data, bytes, getStream(ctx), cnrtMemcpyDevToHost);
    syncStreamInCtx(ctx);
    for (int i = 0; i < len && i < max_len; ++i) {
        std::cout << static_cast<CastT>(reinterpret_cast<RealT*>(ptr.get())[i]) << " ";
    }
    std::cout << std::endl;
}

void printDevData(diopiContextHandle_t ctx, DiopiTensor tensor) {
    int64_t len = tensor.numel();
    void* dataIn = tensor.data();
    int64_t maxLen = 10;
    switch (tensor.dtype()) {
        case diopi_dtype_bool:
            printDevDataInternal<bool, int32_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_uint8:
            printDevDataInternal<uint8_t, int32_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_int8:
            printDevDataInternal<int8_t, int32_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_uint16:
            printDevDataInternal<uint16_t, uint16_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_int16:
            printDevDataInternal<int16_t, int16_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_uint32:
            printDevDataInternal<uint32_t, uint32_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_int32:
            printDevDataInternal<int32_t, int32_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_uint64:
            printDevDataInternal<uint64_t, uint64_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_int64:
            printDevDataInternal<int64_t, int64_t>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_float32:
            printDevDataInternal<float, float>(ctx, dataIn, len, maxLen);
            break;
        case diopi_dtype_float64:
            printDevDataInternal<double, double>(ctx, dataIn, len, maxLen);
            break;
        default:
            std::cout << "unsupported dtype" << std::endl;
            break;
    }
    return;
}

template<typename T>
void printVec(std::vector<T> vec) {
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
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

}  // namespace
}  // namespace camb
}  // namespace impl

#endif  // DIOPI_IMPL_CAMB_COMMON_DEBUG_HPP_
