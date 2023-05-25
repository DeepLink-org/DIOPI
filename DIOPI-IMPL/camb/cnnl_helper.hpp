/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_IMPL_CAMB_CNNL_HELPER_HPP_
#define DIOPI_IMPL_CAMB_CNNL_HELPER_HPP_

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
            setLastErrorString("cnnl error %d : %s at %s:%d", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
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
        const std::vector<int64_t>& dimStride = t.stride();
        size_t dim = dimSize.size();
        std::vector<int32_t> shape(dim);
        std::vector<int32_t> stride(dim);

        cnnlDataType_t dtype;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, t.dtype()));

        if (!dim) {
            std::vector<int> dimArray(1, 1);
            DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(get(), CNNL_LAYOUT_ARRAY, dtype, 1, dimArray.data(), dimArray.data()));
            return diopiSuccess;
        }

        if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC || layout == CNNL_LAYOUT_NLC) {
            shape[0] = dimSize[0];
            stride[0] = dimStride[0];
            for (size_t i = 0; i < dim - 1; ++i) {
                const int index = (i + 1) % (dim - 1) + 1;
                shape[i + 1] = dimSize[index];
                stride[i + 1] = dimStride[index];
            }
        } else if (layout == CNNL_LAYOUT_HWCN) {
            // HWCN is only used by depthwise conv now, and the dim is 4
            DIOPI_CHECK(dim == 4, "depthwise convolution input's dim must be 4!");
            auto convertShapeStrideHwcn = [](const std::vector<int64_t>& vec, std::vector<int>& targetVec) {
                targetVec[0] = static_cast<int>(vec[2]);
                targetVec[1] = static_cast<int>(vec[3]);
                targetVec[2] = static_cast<int>(vec[1]);
                targetVec[3] = static_cast<int>(vec[0]);
            };
            convertShapeStrideHwcn(dimSize, shape);
            convertShapeStrideHwcn(dimStride, stride);
        } else {
            for (size_t i = 0; i < dim; ++i) {
                shape[i] = dimSize[i];
                stride[i] = dimStride[i];
            }
        }

        DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(get(), layout, dtype, shape.size(), shape.data(), stride.data()));
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
        cnnlHandlePool_.emplace(queue, cnnlHandle);
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
    CnnlTransposeDescriptor() = default;

    CnnlTransposeDescriptor(const int dim, const int* permute) { set(dim, permute); }

    diopiError_t set(const int dim, const int* permute) {
        DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(get(), dim, permute));
        return diopiSuccess;
    }
};

class CnnlReduceDescriptor final : public CnnlDescBase<cnnlReduceDescriptor_t, cnnlCreateReduceDescriptor, cnnlDestroyReduceDescriptor> {
public:
    CnnlReduceDescriptor() = default;

    diopiError_t set(DiopiTensor& t, std::vector<int64_t> axis, cnnlReduceOp_t reduceOp, cnnlReduceIndices_t isIndices, cnnlIndicesType_t indicesType,
                     cnnlDataType_t tensorType) {
        int axisNum = axis.size();
        std::vector<int> axisList(axisNum);
        for (int i = 0; i < axisNum; i++) {
            axisList[i] = static_cast<int>(axis[i]);
        }
        DIOPI_CALLCNNL(cnnlSetReduceDescriptor(get(), axisList.data(), axisNum, reduceOp, tensorType, CNNL_NOT_PROPAGATE_NAN, isIndices, indicesType));
        return diopiSuccess;
    }
};

diopiError_t cnnlTranspose(diopiContextHandle_t& ctx, cnnlHandle_t& handle, DiopiTensor& in, DiopiTensor& out, cnnlTensorLayout_t layoutIn,
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

#endif  // DIOPI_IMPL_CAMB_CNNL_HELPER_HPP_
