/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_MLU_OPS_HELPER_HPP_
#define IMPL_CAMB_MLU_OPS_HELPER_HPP_

#include <mlu_op.h>

#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "diopi_helper.hpp"

#define DIOPI_CALLMLUOP(Expr)                                                                                                                     \
    do {                                                                                                                                          \
        mluOpStatus_t ret = Expr;                                                                                                                 \
        if (ret != ::MLUOP_STATUS_SUCCESS) {                                                                                                      \
            impl::camb::setLastErrorString("mlu ops error %d: %s in %s at %s:%d\n", ret, mluOpGetErrorString(ret), __func__, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                                                                                            \
        }                                                                                                                                         \
    } while (false);

#define DIOPI_CHECKMLUOP(Expr)                                                                             \
    do {                                                                                                   \
        mluOpStatus_t ret = Expr;                                                                          \
        if (ret != MLUOP_STATUS_SUCCESS) {                                                                 \
            printf("mlu ops error %d : %s at %s:%d\n", ret, mluOpGetErrorString(ret), __FILE__, __LINE__); \
            std::abort();                                                                                  \
        }                                                                                                  \
    } while (false);

namespace impl {
namespace camb {
class MluOpDataType final {
public:
    static diopiError_t convertToMluOpType(mluOpDataType_t *mluOpType, diopiDtype_t type);
    static bool isFloatPoint(mluOpDataType_t mluOpDT);
    static bool isInteger(mluOpDataType_t mluOpDT);
    static bool isBool(mluOpDataType_t mluOpDT);
};

template <typename T, mluOpStatus_t (*fnCreate)(T *), mluOpStatus_t (*fnDestroy)(T)>
class MluOpResourceGuard final {
public:
    MluOpResourceGuard() { DIOPI_CHECKMLUOP(fnCreate(&resource_)); }

    ~MluOpResourceGuard() { DIOPI_CHECKMLUOP(fnDestroy(resource_)); }

    T &get() { return resource_; }

protected:
    T resource_{0};
};

template <typename T, mluOpStatus_t (*fnCreate)(T *), mluOpStatus_t (*fnDestroy)(T)>
class MluOpDescBase {
public:
    MluOpDescBase() { DIOPI_CHECKMLUOP(fnCreate(&resource_)); }

    virtual ~MluOpDescBase() { DIOPI_CHECKMLUOP(fnDestroy(resource_)); }

    T &get() { return resource_; }

protected:
    T resource_{0};
};

class MluOpTensorDesc : public MluOpDescBase<mluOpTensorDescriptor_t, mluOpCreateTensorDescriptor, mluOpDestroyTensorDescriptor> {
public:
    MluOpTensorDesc() = default;

    template <typename... Args>
    explicit MluOpTensorDesc(Args &&...args) {
        DIOPI_CHECK_ABORT(set(std::forward<Args>(args)...) == diopiSuccess, "%s", "mlu ops failed to set mluOpTensorDescriptor_t object");
    }

    MluOpTensorDesc(const MluOpTensorDesc &other) = delete;
    MluOpTensorDesc(MluOpTensorDesc &&other) = delete;
    MluOpTensorDesc &operator=(const MluOpTensorDesc &other) = delete;

    diopiError_t set(mluOpDataType_t dtype, const std::vector<int32_t> &shape, const std::vector<int32_t> &stride, mluOpTensorLayout_t layout) {
        size_t dim = shape.size();
        std::vector<int32_t> shapeTmp(dim);
        std::vector<int32_t> strideTmp(dim);
        if (!dim) {
            std::vector<int> dimArray = {1};
            DIOPI_CALLMLUOP(mluOpSetTensorDescriptorEx(get(), MLUOP_LAYOUT_ARRAY, dtype, 1, dimArray.data(), dimArray.data()));
            return diopiSuccess;
        }
        if (layout == MLUOP_LAYOUT_NHWC || layout == MLUOP_LAYOUT_NDHWC || layout == MLUOP_LAYOUT_NLC) {
            shapeTmp[0] = shape[0];
            strideTmp[0] = stride[0];
            for (size_t i = 0; i < dim - 1; ++i) {
                const int index = (i + 1) % (dim - 1) + 1;
                shapeTmp[i + 1] = shape[index];
                strideTmp[i + 1] = stride[index];
            }
        } else if (layout == MLUOP_LAYOUT_HWCN) {
            // HWCN is only used by depthwise conv now, and the dim is 4
            DIOPI_CHECK(dim == 4, "depthwise convolution input's dim must be 4!");
            auto convertShapeStrideHwcn = [](const std::vector<int32_t> &vec, std::vector<int32_t> &targetVec) {
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
        DIOPI_CALLMLUOP(mluOpSetTensorDescriptorEx(get(), layout, dtype, shapeTmp.size(), shapeTmp.data(), strideTmp.data()));
        return diopiSuccess;
    }
    template <typename T>
    diopiError_t set(diopiDtype_t dtype, const std::vector<T> &shape, const std::vector<T> &stride, mluOpTensorLayout_t layout) {
        mluOpDataType_t mluOpDtype;
        DIOPI_CALL(MluOpDataType::convertToMluOpType(&mluOpDtype, dtype));
        std::vector<int32_t> shapeTmp(shape.begin(), shape.end());
        std::vector<int32_t> strideTmp(stride.begin(), stride.end());
        DIOPI_CALL(set(mluOpDtype, shapeTmp, strideTmp, layout));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T &t, mluOpTensorLayout_t layout) {
        std::vector<int32_t> shape(t.shape().begin(), t.shape().end());
        std::vector<int32_t> stride(t.stride().begin(), t.stride().end());
        mluOpDataType_t mluOpDtype;
        DIOPI_CALL(MluOpDataType::convertToMluOpType(&mluOpDtype, t.dtype()));
        DIOPI_CALL(set(mluOpDtype, shape, stride, layout));
        return diopiSuccess;
    }

    template <typename T>
    diopiError_t set(T &t, mluOpTensorLayout_t layout, std::vector<int> dims) {
        mluOpDataType_t mluOpDtype;
        DIOPI_CALL(MluOpDataType::convertToMluOpType(&mluOpDtype, t.dtype()));
        DIOPI_CALLMLUOP(mluOpSetTensorDescriptor(get(), layout, mluOpDtype, dims.size(), dims.data()));
        return diopiSuccess;
    }
};

class MluOpHandlePool final {
public:
    mluOpHandle_t insert(cnrtQueue_t queue) {
        assert((mluOpHandlePool_.find(queue) == mluOpHandlePool_.end()) && "The queue inserted exists in the pool");
        std::lock_guard<std::mutex> gurad(mutex_);
        mluOpHandle_t mluOpHandle;
        mluOpCreate(&mluOpHandle);
        mluOpSetQueue(mluOpHandle, queue);
        mluOpHandlePool_.emplace(queue, mluOpHandle);
        return mluOpHandle;
    }

    mluOpHandle_t get(cnrtQueue_t queue) {
        mutex_.lock();
        auto it = mluOpHandlePool_.find(queue);
        mutex_.unlock();
        if (it != mluOpHandlePool_.end()) {
            return it->second;
        } else {
            return insert(queue);
        }
    }

    mluOpHandle_t get(diopiContextHandle_t ctx) {
        cnrtQueue_t queue = getStream(ctx);
        return get(queue);
    }

private:
    std::unordered_map<cnrtQueue_t, mluOpHandle_t> mluOpHandlePool_;
    std::mutex mutex_;
};

extern MluOpHandlePool mluOpHandlePool;

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_MLU_OPS_HELPER_HPP_
