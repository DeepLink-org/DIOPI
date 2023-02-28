#ifndef IMPL_CAMB_CNNL_HELPER_HPP_
#define IMPL_CAMB_CNNL_HELPER_HPP_

#include <cnnl.h>

#include <vector>

#include "diopi_helper.hpp"
#include "error.hpp"

#define DIOPI_CHECK(cond, str)                                                         \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            impl::camb::set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                 \
        }                                                                              \
    } while (false);

#define DIOPI_CALLCNNL(Expr)                                                                                                      \
    do {                                                                                                                          \
        ::cnnlStatus_t ret = Expr;                                                                                                \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                                       \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                            \
        }                                                                                                                         \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                                                                     \
    do {                                                                                                                          \
        ::cnnlStatus_t ret = Expr;                                                                                                \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                                       \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
        }                                                                                                                         \
    } while (false);

template <typename T, ::cnnlStatus_t (*fnCreate)(T*), ::cnnlStatus_t (*fnDestroy)(T)>
class CnnlResourceGuard final {
public:
    CnnlResourceGuard() { DIOPI_CHECKCNNL(fnCreate(&resource_)); }

    ~CnnlResourceGuard() { DIOPI_CHECKCNNL(fnDestroy(resource_)); }

    T& get() { return resource_; }

protected:
    T resource_{0};
};

diopiError_t convertType(cnnlDataType_t* cnnlType, diopiDtype_t type);

class CnnlTensorDesc {
public:
    CnnlTensorDesc() {
        cnnlStatus_t ret = cnnlCreateTensorDescriptor(&desc);
        if (ret != CNNL_STATUS_SUCCESS) {
            impl::camb::set_last_error_string("failed to cnnlCreateTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
        }
    }

    CnnlTensorDesc(auto& t, cnnlTensorLayout_t layout) {
        cnnlStatus_t ret = cnnlCreateTensorDescriptor(&desc);
        if (ret != CNNL_STATUS_SUCCESS) {
            impl::camb::set_last_error_string("failed to cnnlCreateTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
        }
        diopiError_t status = set(t, layout);
        if (ret != CNNL_STATUS_SUCCESS) {
            impl::camb::set_last_error_string("failed to cnnlSetTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
        }
    }

    ~CnnlTensorDesc() {
        cnnlStatus_t ret = cnnlDestroyTensorDescriptor(desc);
        if (ret != CNNL_STATUS_SUCCESS) {
            impl::camb::set_last_error_string("failed to cnnlDestroyTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
        }
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout) {
        int dimNb = t.shape().len;
        auto dimSize = t.shape().data;
        std::vector<int> shape(dimNb);
        for (size_t i = 0; i < dimNb; ++i) {
            shape[i] = dimSize[i];
        }
        DIOPI_CALL(set(t, layout, shape));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout, std::vector<int> dims) {
        cnnlDataType_t dtype;
        DIOPI_CALL(convertType(&dtype, t.dtype()));
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(this->get(), layout, dtype, dims.size(), dims.data()));
        return diopiSuccess;
    }

    cnnlTensorDescriptor_t& get() { return desc; }

protected:
    cnnlTensorDescriptor_t desc{0};
};

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
