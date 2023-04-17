/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_CNNL_HELPER_HPP_
#define IMPL_CAMB_CNNL_HELPER_HPP_

#include <cnnl.h>

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "diopi_helper.hpp"

namespace impl {
namespace camb {

#define DIOPI_CALLCNNL(Expr)                                                                                          \
    do {                                                                                                              \
        ::cnnlStatus_t ret = Expr;                                                                                    \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                           \
            set_last_error_string("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                \
        }                                                                                                             \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                                          \
    do {                                                                                               \
        ::cnnlStatus_t ret = Expr;                                                                     \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                            \
            printf("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
            std::abort();                                                                              \
        }                                                                                              \
    } while (false);

class CnnlDataType final {
public:
    static diopiError_t convertToCnnlType(cnnlDataType_t* cnnlType, diopiDtype_t type);
    static bool isFloatPoint(cnnlDataType_t cnnlDT);
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

template <typename T, ::cnnlStatus_t (*fnCreate)(T*), ::cnnlStatus_t (*fnDestroy)(T)>
class CnnlDescBase {
public:
    CnnlDescBase() { DIOPI_CHECKCNNL(fnCreate(&resource_)); }

    virtual ~CnnlDescBase() { DIOPI_CHECKCNNL(fnDestroy(resource_)); }

    T& get() { return resource_; }

protected:
    T resource_{0};
};

class CnnlTensorDesc : public CnnlDescBase<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> {
public:
    CnnlTensorDesc() = default;

    template <typename... Args>
    explicit CnnlTensorDesc(Args&&... args) {
        DIOPI_CHECK_ABORT(set(std::forward<Args>(args)...) == diopiSuccess, "%s", "cnnl failed to set cnnlTensorDescriptor_t object");
    }

    CnnlTensorDesc(const CnnlTensorDesc& other) = delete;
    CnnlTensorDesc(CnnlTensorDesc&& other) = delete;
    CnnlTensorDesc& operator=(const CnnlTensorDesc& other) = delete;

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout) {
        const std::vector<int64_t>& dimSize = t.shape();
        size_t dim = dimSize.size();
        std::vector<int32_t> shape(dim);

        cnnlDataType_t dtype;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, t.dtype()));

        if (!dim) {
            std::vector<int> dim_array(1, 1);
            DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(get(), CNNL_LAYOUT_ARRAY, dtype, 1, dim_array.data(), dim_array.data()));
            return diopiSuccess;
        }

        if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC || layout == CNNL_LAYOUT_NLC) {
            shape[0] = dimSize[0];
            for (size_t i = 0; i < dim - 1; ++i) {
                shape[i + 1] = dimSize[(i + 1) % (dim - 1) + 1];
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
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(get(), layout, dtype, shape.size(), shape.data()));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout, std::vector<int> dims) {
        cnnlDataType_t dtype;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, t.dtype()));
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(get(), layout, dtype, dims.size(), dims.data()));
        return diopiSuccess;
    }
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

class CnnlTransposeDescriptor final : public CnnlDescBase<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> {
public:
    CnnlTransposeDescriptor() {}

    CnnlTransposeDescriptor(const int dim, const int* permute) { set(dim, permute); }

    diopiError_t set(const int dim, const int* permute) {
        DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(get(), dim, permute));
        return diopiSuccess;
    }
};

class CnnlReduceDescriptor final : public CnnlDescBase<cnnlReduceDescriptor_t, cnnlCreateReduceDescriptor, cnnlDestroyReduceDescriptor> {
public:
    CnnlReduceDescriptor() {}

    diopiError_t set(DiopiTensor& t, std::vector<int64_t> axis, cnnlReduceOp_t reduce_op, cnnlReduceIndices_t is_indices, cnnlIndicesType_t indices_type,
                     cnnlDataType_t tensor_type) {
        int axis_num = axis.size();
        std::vector<int> axis_list(axis_num);
        for (int i = 0; i < axis_num; i++) {
            axis_list[i] = static_cast<int>(axis[i]);
        }
        DIOPI_CALLCNNL(cnnlSetReduceDescriptor(get(), axis_list.data(), axis_num, reduce_op, tensor_type, CNNL_NOT_PROPAGATE_NAN, is_indices, indices_type));
        return diopiSuccess;
    }
};

diopiError_t cnnl_transpose(diopiContextHandle_t& ctx, cnnlHandle_t& handle, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn,
                            cnnlTensorLayout_t layoutOut);

struct HashCnnlCastDType {
    size_t operator()(const std::vector<diopiDtype_t>& vec) const {
        size_t ret = 0;
        for (auto it : vec) {
            ret = (ret ^ static_cast<size_t>(it)) + 0x9e3779b9 + (ret << 6) + (ret >> 2);
        }
        return ret;
    }
};

// global var
extern const std::unordered_map<std::vector<diopiDtype_t>, cnnlCastDataType_t, HashCnnlCastDType> gCnnlCastDataTypeMapping;
extern CnnlHandlePool cnnlHandlePool;

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
