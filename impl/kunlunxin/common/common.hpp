#ifndef IMPL_KLX_PYTORCH_COMMON_COMMON_HPP_
#define IMPL_KLX_PYTORCH_COMMON_COMMON_HPP_

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

#define DIOPI_CALL_XDNN(Expr)                                                                                                                       \
    do {                                                                                                                                            \
        int ret = Expr;                                                                                                                             \
        if (static_cast<xdnn::Error_t>(ret) != xdnn::SUCCESS) {                                                                                     \
            set_last_error_string(                                                                                                                  \
                "call a klxrt function failed: (%s), return code=%s, %s at %s:%d\n", #Expr, XdnnGetErrorString(ret), __func__, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                                              \
        }                                                                                                                                           \
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
        if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU1) {
            std::cout << "running in KunLun1" << std::endl;
        } else if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU2) {
            std::cout << "running in KunLun2" << std::endl;
        } else if (tls_raw_ctx_.get()->dev().type() == xdnn::kXPU3) {
            std::cout << "running in KunLun3" << std::endl;
        } else {
            std::cout << "running in unknown XPU device: " << static_cast<int>(tls_raw_ctx_.get()->dev().type()) << std::endl;
        }
        std::cout << "thread 0x" << std::hex << std::this_thread::get_id() << " set context xpu stream: " << pstream << std::endl;
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

//#define DEBUG false
#define DEBUG true

inline bool isInt(const diopiScalar_t* scalar) { return scalar->stype <= 7; }

inline bool isFloat(const diopiScalar_t* scalar) { return scalar->stype > 7; }

xdnn_pytorch::Scalar build_xtorch_scalar(const diopiScalar_t* scalar) {
    if (scalar == nullptr) {
        NOT_SUPPORTED("scalar is null ptr, we use temporarily zero");
        return xdnn_pytorch::Scalar({});
    }
    if (DEBUG) {
        printf("scalar type is %d\n", (int)scalar->stype);
    }
    // Note: indicate explictly the return type to make correctly at::Scalar.
    xdnn_pytorch::Scalar temp;
    if (isInt(scalar)) {
        int64_t ival = scalar->ival;
        temp.type = xdnn_pytorch::ScalarType::kint64;
        *((int64_t*)(&temp.data)) = ival;
        if (DEBUG) {
            printf("scalar value is %ld\n", scalar->ival);
        }
    } else {
        float fval = scalar->fval;
        temp.type = xdnn_pytorch::ScalarType::kfloat32;
        *((float*)(&temp.data)) = fval;
        if (DEBUG) {
            printf("scalar value is %f\n", fval);
        }
    }
    return temp;
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
        // TODO:
        /*
        case diopi_dtype_float64:
            return xdnn_pytorch::ScalarType::kfloat64;
        */
        default:
            // NOT_SUPPORTED("diopi dytpe");
            printf("NOT SUPPORTED diopi dytpe: %d\n", (int)dt);
            return xdnn_pytorch::ScalarType::UnknownScalarType;
    }
}

xtorch_vec build_xtorch_vec(diopiSize_t size) {
    xdnn_pytorch::IntArrayRef tmp(size.data, size.len);
    xtorch_vec res;
    IntArrayRef2IntVector(tmp, res);
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
        return {{0}, {0}, xdnn_pytorch::ScalarType::kfloat32, (void*)nullptr};
    }

    // NOTE: diopiTensor default construct func, storage_ is nullptr.
    // if calling data(), Segmentation fault
    // example: diopi_test/python/tests/test_last_error.py, diopiTensor()
    diopiSize_t dsize;
    diopiGetTensorShape(tensor, &dsize);
    if (DEBUG) {
        printf("tensor dsize len is %ld\n", dsize.len);
    }
    if (dsize.len == 0) {
        NOT_SUPPORTED("tensor is empty");
        return {{0}, {0}, xdnn_pytorch::ScalarType::kfloat32, (void*)nullptr};
    }

    void* data = nullptr;
    diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);
    if (DEBUG) {
        printf("tensor ptr is %p\n", data);
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    if (DEBUG) {
        printf("tensor dtype is %d\n", (int)dtype);
    }
    xdnn_pytorch::ScalarType scalar_type = get_xtorch_type(dtype);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    std::vector<int64_t> xtorch_tensor_sizes(shape.data, shape.data + shape.len);
    if (DEBUG) {
        printf("tensor shape is ");
        for (auto i : xtorch_tensor_sizes) {
            printf("%ld ", i);
        }
        printf("\n");
    }

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    std::vector<int64_t> xtorch_tensor_strides(stride.data, stride.data + stride.len);
    if (DEBUG) {
        printf("tensor stride is ");
        for (auto i : xtorch_tensor_strides) {
            printf("%ld ", i);
        }
        printf("\n");
    }
    return {xtorch_tensor_sizes, xtorch_tensor_strides, scalar_type, (void*)data};
}

template <typename T>
decltype(auto) build_xtorch_tensorlist(T* tensors, int64_t numTensors) {
    std::vector<xdnn_pytorch::Tensor> vecAtTensor;
    for (size_t i = 0; i < numTensors; ++i) {
        vecAtTensor.emplace_back(build_xtorch_tensor(tensors[i]));
    }
    return vecAtTensor;
}

}  // namespace kunlunxin
}  // namespace impl

#endif  //  IMPL_KLX_PYTORCH_COMMON_COMMON_HPP_
