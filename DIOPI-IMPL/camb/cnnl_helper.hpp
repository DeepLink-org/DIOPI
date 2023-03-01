#ifndef IMPL_CAMB_CNNL_HELPER_HPP_
#define IMPL_CAMB_CNNL_HELPER_HPP_

#include <cnnl.h>

#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>
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
        const std::vector<int32_t>& shape = t.shape();
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

class CnnlHandlePool final {
public:
    cnnlHandle_t insert(cnrtQueue_t queue) {
        assert((cnnlHandlePool_.find(queue) == cnnlHandlePool_.end()) && "The queue inserted exists in the pool");
        std::lock_guard<std::mutex> gurad(mutex_);
        cnnlHandle_t cnnlHandle;
        cnnlCreate(&cnnlHandle);
        cnnlSetQueue(cnnlHandle, queue);
        cnnlHandlePool_.emplace(std::make_pair(queue, cnnlHandle));
        return cnnlHandle;
    }

    cnnlHandle_t get(cnrtQueue_t queue) {
        mutex_.lock();
        auto it = cnnlHandlePool_.find(queue);
        mutex_.unlock();
        if (it != cnnlHandlePool_.end()) {
            return it->second;
        } else {
            return insert(queue);
        }
    }
    cnnlHandle_t get(diopiContextHandle_t ctx) {
        cnrtQueue_t queue = impl::camb::getStream(ctx);
        return get(queue);
    }

private:
    std::unordered_map<cnrtQueue_t, cnnlHandle_t> cnnlHandlePool_;
    std::mutex mutex_;
};

extern CnnlHandlePool cnnlHandlePool;

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
