#ifndef IMPL_CAMB_CNNL_HELPER_HPP_
#define IMPL_CAMB_CNNL_HELPER_HPP_

#include <cnnl.h>

#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "diopi_helper.hpp"


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
    CnnlTensorDesc() {}

    CnnlTensorDesc(auto& t, cnnlTensorLayout_t layout) {
        diopiError_t status = set(t, layout);
        if (status != diopiSuccess) {
            impl::camb::set_last_error_string("failed to cnnlSetTensorDescriptor %d at %s:%d", status, __FILE__, __LINE__);
        }
    }

    ~CnnlTensorDesc() {
        if (desc != nullptr) {
            cnnlStatus_t ret = cnnlDestroyTensorDescriptor(desc);
            if (ret != CNNL_STATUS_SUCCESS) {
                impl::camb::set_last_error_string("failed to cnnlDestroyTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
            }
        }
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout) {
        const std::vector<int32_t>& dimSize = t.shape();
        int dim = dimSize.size();
        std::vector<int32_t> shape(dim);

        if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC
                || layout == CNNL_LAYOUT_NLC) {
            shape[0] = dimSize[0];
            for (size_t i = 0; i < dim - 1; ++i) {
                shape[i+1] = dimSize[(i + 1) % (dim - 1) + 1];
            }
        } else if (layout == CNNL_LAYOUT_HWCN) {
            // HWCN is only used by depthwise conv now, and the dim is 4
            DIOPI_CHECK(dim == 4, "depthwise convolution input's dim must be 4!");
            shape[0] = dimSize[2];
            shape[1] = dimSize[3];
            shape[2] = dimSize[1];
            shape[3] = dimSize[0];
        } else {
            for (size_t i = 0; i < dim; ++i) {
                shape[i] = dimSize[i];
            }
        }

        DIOPI_CALL(set(t, layout, shape));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout, std::vector<int> dims) {
        cnnlDataType_t dtype;
        DIOPI_CALL(convertType(&dtype, t.dtype()));
        DIOPI_CALLCNNL(cnnlCreateTensorDescriptor(&desc));
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

template <typename T, ::cnnlStatus_t (*fnCreate)(T*), ::cnnlStatus_t (*fnDestroy)(T)>
class CnnlDescBase {
public:
    CnnlDescBase() { DIOPI_CHECKCNNL(fnCreate(&resource_)); }

    ~CnnlDescBase() { DIOPI_CHECKCNNL(fnDestroy(resource_)); }

    T& get() { return resource_; }

protected:
    T resource_{0};
};

class CnnlTransposeDescriptor final
    : public CnnlDescBase<cnnlTransposeDescriptor_t,
          cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> {
public:
    CnnlTransposeDescriptor() {}

    CnnlTransposeDescriptor(const int dim, const int* permute) {
        set(dim, permute);
    }

    diopiError_t set(const int dim, const int* permute) {
        DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(get(), dim, permute));
        return diopiSuccess;
    }
};

extern CnnlHandlePool cnnlHandlePool;

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
