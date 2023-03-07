#ifndef IMPL_CAMB_CNNL_HELPER_HPP_
#define IMPL_CAMB_CNNL_HELPER_HPP_

#include <cnnl.h>

#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "diopi_helper.hpp"

namespace impl {
namespace camb {

#define DIOPI_CALLCNNL(Expr)                                                                                                      \
    do {                                                                                                                          \
        ::cnnlStatus_t ret = Expr;                                                                                                \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                                       \
            set_last_error_string("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                            \
        }                                                                                                                         \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                                                                     \
    do {                                                                                                                          \
        ::cnnlStatus_t ret = Expr;                                                                                                \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                                       \
            set_last_error_string("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
        }                                                                                                                         \
    } while (false);

class CnnlDataType final {
public:
    static diopiError_t convertToCnnlType(cnnlDataType_t* cnnlType, diopiDtype_t type);
    static bool isFloat(cnnlDataType_t cnnlDT);
    static bool isInteger(cnnlDataType_t cnnlDT);
    static bool isBool(cnnlDataType_t cnnlDT);
};

template <typename T, ::cnnlStatus_t (*fnCreate)(T*), ::cnnlStatus_t (*fnDestroy)(T)>
class CnnlResourceGuard final {
public:
    CnnlResourceGuard() { DIOPI_CHECKCNNL(fnCreate(&resource_)); }

    ~CnnlResourceGuard() { DIOPI_CHECKCNNL(fnDestroy(resource_)); }

    T& get() { return resource_; }

protected:
    T resource_{0};
};

class CnnlTensorDesc {
public:
    CnnlTensorDesc() {}

    CnnlTensorDesc(auto& t, cnnlTensorLayout_t layout) {
        diopiError_t status = set(t, layout);
        if (status != diopiSuccess) {
            set_last_error_string("failed to cnnlSetTensorDescriptor %d at %s:%d", status, __FILE__, __LINE__);
        }
    }

    ~CnnlTensorDesc() {
        if (desc != nullptr) {
            cnnlStatus_t ret = cnnlDestroyTensorDescriptor(desc);
            if (ret != CNNL_STATUS_SUCCESS) {
                set_last_error_string("failed to cnnlDestroyTensorDescriptor %d at %s:%d", ret, __FILE__, __LINE__);
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
        DIOPI_CALLCNNL(cnnlCreateTensorDescriptor(&desc));
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, t.dtype()));
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(desc, layout, dtype, dims.size(), dims.data()));
        return diopiSuccess;
    }

    cnnlTensorDescriptor_t get() { return desc; }

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
        cnrtQueue_t queue = getStream(ctx);
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

class CnnlReduceDescriptor final
    : public CnnlDescBase<cnnlReduceDescriptor_t,
          cnnlCreateReduceDescriptor, cnnlDestroyReduceDescriptor> {
public:
    CnnlReduceDescriptor() {}

    CnnlReduceDescriptor(auto& t, diopiSize_t dim,
                         const cnnlReduceOp_t reduceOp,
                         cnnlDataType_t dtype,
                         cnnlNanPropagation_t nanPropagation,
                         cnnlReduceIndices_t indices,
                         cnnlIndicesType_t indicesType) {
        set(t, dim, reduceOp, dtype, nanPropagation, indices, indicesType);
    }

    template <typename T>
    diopiError_t set(T& t, diopiSize_t dim,
            const cnnlReduceOp_t reduceOp,
            cnnlDataType_t dtype,
            cnnlNanPropagation_t nanPropagation,
            cnnlReduceIndices_t indices,
            cnnlIndicesType_t indicesType) {
        /* if dim is not set, all dimensions are reduced. */
        std::vector<int> axis;
        if (dim.len > 0) {
            std::vector<int> dims{dim.data, dim.data + dim.len};
            axis = dims;
        } else {
            int input_size = t.shape().size();
            for (int i = 0; i < input_size; i++) {
                axis.push_back(i);
            }
        }
        DIOPI_CALLCNNL(cnnlSetReduceDescriptor(
            get(), axis.data(), axis.size(), reduceOp, dtype,
            nanPropagation, indices, indicesType));

        return diopiSuccess;
    }
};

template<typename T>
diopiError_t cnnl_transpose(diopiContextHandle_t& ctx,
                            cnnlHandle_t& handle, T& in,
                            DiopiTensor<diopiTensorHandle_t>& out,
                            cnnlTensorLayout_t layoutIn,
                            cnnlTensorLayout_t layoutOut) {
    std::vector<int> order;
    if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {1, 2, 3, 0};
    } else if (layoutIn == CNNL_LAYOUT_NHWC && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {0, 3, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_HWCN) {
        order = {2, 3, 1, 0};
    } else if (layoutIn == CNNL_LAYOUT_NCHW && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {0, 2, 3, 1};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NHWC) {
        order = {3, 0, 1, 2};
    } else if (layoutIn == CNNL_LAYOUT_HWCN && layoutOut == CNNL_LAYOUT_NCHW) {
        order = {3, 2, 0, 1};
    } else {
        set_last_error_string("unkown layout error, layout should be"
        "in [CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_HWCN], at %s:%s", __FILE__, __LINE__);
        return diopiDtypeNotSupported;
    }
    CnnlTensorDesc inDesc(in, layoutIn);
    CnnlTensorDesc outDesc(out, layoutOut);
    CnnlTransposeDescriptor transDesc(order.size(), order.data());
    size_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetTransposeWorkspaceSize(handle, inDesc.get(), transDesc.get(), &workspace_size));

    void* workspace_ptr = workspace_size== 0 ? requiresBuffer(ctx, workspace_size).data() : nullptr;
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inDesc.get(),
                                      in.data(), outDesc.get(), out.data(),
                                      workspace_ptr, workspace_size));
    return diopiSuccess;
}

extern CnnlHandlePool cnnlHandlePool;

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
