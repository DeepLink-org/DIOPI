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

extern bool isRecordOn;

#define DIOPI_CALLCNNL(Expr)                                                                                                                    \
    do {                                                                                                                                        \
        void* record = nullptr;                                                                                                                 \
        if (isRecordOn) {                                                                                                                       \
            DIOPI_RECORD_START(Expr);                                                                                                           \
        }                                                                                                                                       \
        ::cnnlStatus_t ret = Expr;                                                                                                              \
        if (isRecordOn) {                                                                                                                       \
            DIOPI_RECORD_END;                                                                                                                   \
        }                                                                                                                                       \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                                                                     \
            impl::camb::setLastErrorString("cnnl error %d: %s in %s at %s:%d\n", ret, ::cnnlGetErrorString(ret), __func__, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                                          \
        }                                                                                                                                       \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                                            \
    do {                                                                                                 \
        ::cnnlStatus_t ret = Expr;                                                                       \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                                              \
            printf("cnnl error %d : %s at %s:%d\n", ret, ::cnnlGetErrorString(ret), __FILE__, __LINE__); \
            std::abort();                                                                                \
        }                                                                                                \
    } while (false);

namespace impl {
namespace camb {

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
    diopiError_t set(cnnlDataType_t dtype, const std::vector<int32_t>& shape, const std::vector<int32_t>& stride, cnnlTensorLayout_t layout) {
        size_t dim = shape.size();
        std::vector<int32_t> shapeTmp(dim);
        std::vector<int32_t> strideTmp(dim);
        if (!dim) {
            std::vector<int> dimArray(1, 1);
            DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(get(), CNNL_LAYOUT_ARRAY, dtype, 1, dimArray.data(), dimArray.data()));
            return diopiSuccess;
        }
        if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC || layout == CNNL_LAYOUT_NLC) {
            shapeTmp[0] = shape[0];
            strideTmp[0] = stride[0];
            for (size_t i = 0; i < dim - 1; ++i) {
                const int index = (i + 1) % (dim - 1) + 1;
                shapeTmp[i + 1] = shape[index];
                strideTmp[i + 1] = stride[index];
            }
        } else if (layout == CNNL_LAYOUT_HWCN) {
            // HWCN is only used by depthwise conv now, and the dim is 4
            DIOPI_CHECK(dim == 4, "depthwise convolution input's dim must be 4!");
            auto convertShapeStrideHwcn = [](const std::vector<int32_t>& vec, std::vector<int32_t>& targetVec) {
                targetVec[0] = vec[2];
                targetVec[1] = vec[3];
                targetVec[2] = vec[1];
                targetVec[3] = vec[0];
            };
            convertShapeStrideHwcn(shape, shapeTmp);
            convertShapeStrideHwcn(stride, strideTmp);
        } else {
            shapeTmp = shape;
            strideTmp = stride;
        }
        DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(get(), layout, dtype, shapeTmp.size(), shapeTmp.data(), strideTmp.data()));
        return diopiSuccess;
    }
    template <typename T>
    diopiError_t set(diopiDtype_t dtype, const std::vector<T>& shape, const std::vector<T>& stride, cnnlTensorLayout_t layout) {
        cnnlDataType_t cnnlDtype;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&cnnlDtype, dtype));
        std::vector<int32_t> shapeTmp(shape.begin(), shape.end());
        std::vector<int32_t> strideTmp(stride.begin(), stride.end());
        DIOPI_CALL(set(cnnlDtype, shapeTmp, strideTmp, layout));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T& t, cnnlTensorLayout_t layout) {
        std::vector<int32_t> shape(t.shape().begin(), t.shape().end());
        std::vector<int32_t> stride(t.stride().begin(), t.stride().end());
        cnnlDataType_t dtype;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, t.dtype()));
        DIOPI_CALL(set(dtype, shape, stride, layout));
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

class CnnlInterpDescriptor final : public CnnlDescBase<cnnlInterpDescriptor_t, cnnlCreateInterpDescriptor, cnnlDestroyInterpDescriptor> {
public:
    CnnlInterpDescriptor() = default;

    diopiError_t set(cnnlTensorDescriptor_t inputDesc, const cnnlInterpMode_t mode, const cnnlInterpCoordinateTransformationMode_t coordinateTransMode,
                     float* scales) {
        DIOPI_CALLCNNL(cnnlSetInterpDescriptor(this->get(), mode, coordinateTransMode));
        cnnlInterpRoundMode_t roundMode = CNNL_INTERP_FLOOR;
        DIOPI_CALLCNNL(cnnlSetInterpDescriptorEx(this->get(), inputDesc, roundMode, scales, nullptr, -0.75, false));
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

#endif  // IMPL_CAMB_CNNL_HELPER_HPP_
