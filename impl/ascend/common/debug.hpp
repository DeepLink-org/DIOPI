/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_DEBUG_HPP_
#define IMPL_ASCEND_COMMON_DEBUG_HPP_

#include <algorithm>
#include <sstream>
#include <string>

#include "../ascend_tensor.hpp"
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

inline std::string dumpTensor(diopiConstTensorHandle_t th, const std::string& message = "") {
    std::stringstream stream;
    stream << "Tensor(handle:" << th << " " << message;
    if (th) {
        diopiSize_t shape;
        diopiSize_t stride;
        const void* ptr;
        diopiDtype_t dtype;
        diopiDevice_t device;
        diopiGetTensorDtype(th, &dtype);
        diopiGetTensorDataConst(th, &ptr);
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        diopiGetTensorDevice(th, &device);
        stream << " ,data:" << ptr;
        stream << " ,dtype:" << dtype;
        stream << " ,device:" << device;
        stream << " ,shape:";
        std::for_each(shape.data, shape.data + shape.len, [&stream](int64_t v) { stream << v << " "; });
        stream << " ,stride:";
        std::for_each(stride.data, stride.data + stride.len, [&stream](int64_t v) { stream << v << " "; });
    }
    stream << ")";
    return stream.str();
}

inline std::string dumpTensor(const AscendTensor& at, const std::string& message = "") {
    std::stringstream stream;
    stream << "AscendTensor(handle:" << at.data() << " " << message;
    if (at.defined()) {
        auto shape = at.shape();
        auto stride = at.stride();
        stream << " ,data:" << at.data();
        stream << " ,dtype:" << at.dtype();
        stream << " ,device:" << at.device();
        stream << " ,shape:";
        std::for_each(shape.begin(), shape.end(), [&stream](int64_t v) { stream << v << " "; });
        stream << " ,stride:";
        std::for_each(stride.begin(), stride.end(), [&stream](int64_t v) { stream << v << " "; });
    }
    stream << ")";
    return stream.str();
}

void printContiguousTensor(diopiContextHandle_t ctx, const AscendTensor& at, char* name);
void printTensor(diopiContextHandle_t ctx, diopiSize_t& th, char* name);
}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_DEBUG_HPP_
