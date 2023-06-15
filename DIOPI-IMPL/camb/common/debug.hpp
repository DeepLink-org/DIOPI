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
#include <string>
#include <vector>

#include "../diopi_helper.hpp"
#include "float16.hpp"

namespace impl {
namespace camb {

// print the data on dev
template <typename RealT, typename CastT>
void printDevDataInternal(diopiContextHandle_t ctx, void* data, int64_t len, int64_t maxLen) {
    int bytes = sizeof(RealT) * len;
    std::unique_ptr<char> ptr(new char[bytes]);
    std::cout << "data address:" << data << std::endl;
    cnrtMemcpyAsync(ptr.get(), data, bytes, getStream(ctx), cnrtMemcpyDevToHost);
    syncStreamInCtx(ctx);
    std::cout << "[";
    for (int i = 0; i < len && i < maxLen; ++i) {
        std::cout << static_cast<CastT>(reinterpret_cast<RealT*>(ptr.get())[i]) << " ";
    }
    std::cout << "]" << std::endl;
}

inline void printDevData(diopiContextHandle_t ctx, DiopiTensor tensor, std::string name = "name") {
    if (!tensor.defined()) {
        std::cout << "Tensor " << name << " is not defined. Please check it before using `printDevData`." << std::endl;
        return;
    }
    int64_t len = tensor.numel();
    void* dataIn = tensor.data();
    int64_t maxLen = 20;
    int dim = tensor.dim();
    std::cout << "DiopiTensor[" << name << "]: dim" << dim << ", dtype: " << DiopiDataType::dataTypeStr(tensor.dtype()) << ", shape: [";
    for (size_t i = 0; i < dim; i++) {
        std::cout << tensor.shape()[i] << ", ";
    }
    std::cout << "], stride: [";
    for (size_t i = 0; i < dim; i++) {
        std::cout << tensor.stride()[i] << ", ";
    }
    std::cout << "], is_contiguous: " << tensor.isContiguous();
    std::cout << ", is_contiguous(channelsLast): " << tensor.isContiguous(MemoryFormat::ChannelsLast) << std::endl;
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
        case diopi_dtype_float16:
            printDevDataInternal<half_float::half, float>(ctx, dataIn, len, maxLen);
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

template <typename T>
void printVec(const std::string& str, std::vector<T> vec) {
    std::cout << str << ": ";
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

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

}  // namespace camb
}  // namespace impl

#endif  // DIOPI_IMPL_CAMB_COMMON_DEBUG_HPP_
