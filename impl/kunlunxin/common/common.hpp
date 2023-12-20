#ifndef IMPL_KUNLUNXIN_COMMON_COMMON_HPP_
#define IMPL_KUNLUNXIN_COMMON_COMMON_HPP_

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <cassert>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../error.hpp"
#include "xdnn_pytorch/xdnn_pytorch.h"
#include "xpu/runtime.h"

#define DEBUG false
// #define DEBUG true
namespace impl {
namespace kunlunxin {

typedef std::vector<int64_t> xtorch_vec;
// diopi context
static thread_local diopiContextHandle_t context = nullptr;
// xpu context
static thread_local std::shared_ptr<xdnn::Context> tls_raw_ctx_{nullptr};

const char* XdnnGetErrorString(int error) {
    xdnn::Error_t xdnn_error = static_cast<xdnn::Error_t>(error);
    switch (xdnn_error) {
        case xdnn::SUCCESS:
            return "SUCCESS";
        case xdnn::INVALID_PARAM:
            return "INVALID_PARAM";
        case xdnn::RUNTIME_ERROR:
            return "RUNTIME_ERROR";
        case xdnn::NO_ENOUGH_WORKSPACE:
            return "NO_ENOUGH_WORKSPACE";
        case xdnn::NOT_IMPLEMENT:
            return "NOT_IMPLEMENT";
        default:
            return "NOT SUPPORTED xdnn::Error_t";
    }
}

#define DIOPI_CALL_XDNN(Expr)                                                                                                                               \
    do {                                                                                                                                                    \
        int ret = Expr;                                                                                                                                     \
        if (static_cast<xdnn::Error_t>(ret) != xdnn::SUCCESS) {                                                                                             \
            set_last_error_string(                                                                                                                          \
                "call a klxrt function failed: (%s), return code=%d:%s, %s at %s:%d\n", #Expr, ret, XdnnGetErrorString(ret), __func__, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                                                      \
        }                                                                                                                                                   \
    } while (false);

#define WITH_ERROR_CHECK(expr, msg) \
    do {                            \
        diopiError_t err = (expr);  \
        if (err != diopiSuccess) {  \
            throw(msg);             \
        }                           \
    } while (false);

#define DIOPI_KLX_CHECK(condition)                                                                                      \
    do {                                                                                                                \
        if (!(condition)) {                                                                                             \
            set_last_error_string("klx error occurred: (%s), %s at %s:%d\n", #condition, __func__, __FILE__, __LINE__); \
        }                                                                                                               \
    } while (false);

#define NOT_SUPPORTED(str) set_last_error_string("NotSupported: %s, %s at %s:%d", str, __func__, __FILE__, __LINE__);

static xdnn::Context* get_raw_context(bool update_context) {
    if (tls_raw_ctx_.get() == nullptr || update_context) {
        tls_raw_ctx_.reset(xdnn::create_context(), xdnn::destroy_context);
        assert(tls_raw_ctx_.get() != nullptr);
        diopiStreamHandle_t pstream = nullptr;
        diopiGetStream(context, &pstream);
        assert(pstream != nullptr);
        tls_raw_ctx_.get()->xpu_stream = (XPUStream)pstream;
        xdnn::DeviceType dev_type = tls_raw_ctx_.get()->dev().type();
        if (DEBUG) {
            if (dev_type == xdnn::kXPU1) {
                std::cout << "running in KunLun1" << std::endl;
            } else if (dev_type == xdnn::kXPU2) {
                std::cout << "running in KunLun2" << std::endl;
            } else if (dev_type == xdnn::kXPU3) {
                std::cout << "running in KunLun3" << std::endl;
            } else {
                std::cout << "running in unknown XPU device: " << static_cast<int>(tls_raw_ctx_.get()->dev().type()) << std::endl;
            }
            std::cout << "thread 0x" << std::hex << std::this_thread::get_id() << " set context xpu stream: " << pstream << std::endl;
        }
    }
    return tls_raw_ctx_.get();
}

inline xdnn::Context* set_cur_ctx(diopiContextHandle_t ctx) {
    bool update_context = (context != ctx);
    if (update_context) {
        context = ctx;
    }
    return get_raw_context(update_context);
}

inline bool isInt(const diopiScalar_t* scalar) { return scalar->stype <= 7; }

inline bool isFloat(const diopiScalar_t* scalar) { return scalar->stype > 7; }

xdnn_pytorch::Scalar build_xtorch_scalar(const diopiScalar_t* scalar) {
    if (scalar == nullptr) {
        NOT_SUPPORTED("scalar is null ptr, we use temporarily zero");
        return xdnn_pytorch::Scalar({});
    }
    if (DEBUG) {
        printf("scalar type is %d\n", static_cast<int>(scalar->stype));
    }
    // Note: indicate explictly the return type to make correctly at::Scalar.
    if (isInt(scalar)) {
        int64_t ival = scalar->ival;
        return xdnn_pytorch::Scalar::create(ival);
    } else {
        float fval = scalar->fval;
        return xdnn_pytorch::Scalar::create(fval);
    }
}

xdnn_pytorch::ScalarType get_xtorch_type(diopiDtype_t dt) {
    switch (dt) {
        case diopi_dtype_bool:
            return xdnn_pytorch::ScalarType::kbool;
        case diopi_dtype_uint8:
            return xdnn_pytorch::ScalarType::kuint8;
        case diopi_dtype_int8:
            return xdnn_pytorch::ScalarType::kint8;
        case diopi_dtype_int16:
            return xdnn_pytorch::ScalarType::kint16;
        case diopi_dtype_int32:
            return xdnn_pytorch::ScalarType::kint32;
        case diopi_dtype_int64:
            return xdnn_pytorch::ScalarType::kint64;
        case diopi_dtype_float32:
            return xdnn_pytorch::ScalarType::kfloat32;
        case diopi_dtype_float16:
            return xdnn_pytorch::ScalarType::kfloat16;
        case diopi_dtype_bfloat16:
            return xdnn_pytorch::ScalarType::kbfloat16;
        case diopi_dtype_float64:
            return xdnn_pytorch::ScalarType::kfloat64;
        default:
            // NOT_SUPPORTED("diopi dytpe");
            printf("NOT SUPPORTED diopi dytpe: %d\n", static_cast<int>(dt));
            return xdnn_pytorch::ScalarType::UnknownScalarType;
    }
}

xtorch_vec build_xtorch_vec(diopiSize_t size) {
    // xdnn_pytorch::IntArrayRef tmp(size.data, size.len);
    xtorch_vec res;

    // IntArrayRef2IntVector(tmp, res);
    int64_t length = size.len;
    for (int i = 0; i < length; i++) {
        if (i > 100) throw "bad size";
        res.push_back(size.data[i]);
    }
    return res;
}

template <typename T>
xdnn_pytorch::Tensor build_xtorch_tensor(T tensor) {
    if (DEBUG) {
        printf("tensor building... \n");
    }
    if (tensor == nullptr) {
        if (DEBUG) {
            printf("tensor is nullptr\n");
        }
        return {{0}, {0}, xdnn_pytorch::ScalarType::kfloat32, nullptr};
    }

    // NOTE: diopiTensor default construct func, storage_ is nullptr.
    // if calling data(), Segmentation fault
    // example: diopi_test/python/tests/test_last_error.py, diopiTensor()
    diopiSize_t dsize;
    diopiError_t err = diopiGetTensorShape(tensor, &dsize);
    if (err != diopiSuccess) {
        throw "bad tensor shape";
    }
    if (DEBUG) {
        printf("tensor dsize len is %ld\n", dsize.len);
    }

    void* data = nullptr;
    err = diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);
    if (err != diopiSuccess) {
        throw "bad tensor data";
    }
    if (DEBUG) {
        printf("tensor ptr is %p\n", data);
    }

    diopiDtype_t dtype;
    err = diopiGetTensorDtype(tensor, &dtype);
    if (err != diopiSuccess) {
        throw "bad tensor datatype";
    }
    if (DEBUG) {
        printf("tensor dtype is %d\n", static_cast<int>(dtype));
    }
    xdnn_pytorch::ScalarType scalar_type = get_xtorch_type(dtype);

    diopiSize_t shape;
    err = diopiGetTensorShape(tensor, &shape);
    if (err != diopiSuccess) {
        throw "bad tensor shape";
    }
    std::vector<int64_t> xtorch_tensor_sizes(shape.data, shape.data + shape.len);
    if (dsize.len == 0) {  // fill shape for scalar tensor
        xtorch_tensor_sizes.push_back(1);
    }
    if (DEBUG) {
        printf("tensor shape is ");
        for (auto i : xtorch_tensor_sizes) {
            printf("%ld ", i);
        }
        printf("\n");
    }

    diopiSize_t stride;
    err = diopiGetTensorStride(tensor, &stride);
    if (err != diopiSuccess) {
        throw "bad tensor stride";
    }
    std::vector<int64_t> xtorch_tensor_strides(stride.data, stride.data + stride.len);
    if (dsize.len == 0) {  // fill stride for scalar tensor
        xtorch_tensor_strides.push_back(1);
    }
    if (DEBUG) {
        printf("tensor stride is ");
        for (auto i : xtorch_tensor_strides) {
            printf("%ld ", i);
        }
        printf("\n");
    }
    return {xtorch_tensor_sizes, xtorch_tensor_strides, scalar_type, reinterpret_cast<void*>(data)};
}

template <typename T>
decltype(auto) build_xtorch_tensorlist(T* tensors, int64_t numTensors) {
    std::vector<xdnn_pytorch::Tensor> vecAtTensor(numTensors);
    for (size_t i = 0; i < numTensors; ++i) {
        vecAtTensor[i] = build_xtorch_tensor(tensors[i]);
    }
    return vecAtTensor;
}

}  // namespace kunlunxin
}  // namespace impl

#endif  //  IMPL_KUNLUNXIN_COMMON_COMMON_HPP_
