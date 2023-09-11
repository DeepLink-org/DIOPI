/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "debug.hpp"

#include <diopi/diopirt.h>

#include <algorithm>
#include <sstream>

#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

std::string dumpTensor(diopiConstTensorHandle_t th, const std::string& msg) {
    std::stringstream stream;
    if (!msg.empty()) {
        stream << msg.c_str() << "\n";
    }
    stream << "Tensor(handle:" << th;
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

std::string dumpTensor(const AscendTensor& th, const std::string& msg) {
    std::stringstream stream;
    if (!msg.empty()) {
        stream << msg.c_str() << "\n";
    }
    stream << "AscendTensor(handle:" << th.defined();
    if (th.defined()) {
        stream << " ,data:" << th.data();
        stream << " ,dtype:" << th.dtype();
        stream << " ,device:" << th.device();
        stream << " ,shape:";
        std::for_each(th.shape().data(), th.shape().data() + th.dim(), [&stream](int64_t v) { stream << v << " "; });
        stream << " ,stride:";
        std::for_each(th.stride().data(), th.stride().data() + th.dim(), [&stream](int64_t v) { stream << v << " "; });
    }
    stream << ")";
    return stream.str();
}

}  // namespace ascend
}  // namespace impl
