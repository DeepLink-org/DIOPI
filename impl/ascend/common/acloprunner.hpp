#ifndef IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
#define IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <diopi/functions.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace impl {
namespace ascend {

#define TRACK_ACL(x)                                                    \
    {                                                                   \
        static bool enable = std::getenv("DIOPI_TRACK_ACL") != nullptr; \
        if (enable) {                                                   \
            printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);             \
        }                                                               \
    }

#define CALL_ACLRT(Expr)                                                                          \
    {                                                                                             \
        TRACK_ACL(#Expr);                                                                         \
        ::aclError ret = Expr;                                                                    \
        if (ret != ::ACL_SUCCESS) {                                                               \
            throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                                                         \
    }

#define warning(...)                             \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

#define info(...)                                \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

#define check_args(condition, ...)                   \
    if (!(condition)) {                              \
        printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
        printf(__VA_ARGS__);                         \
        printf("\n");                                \
        std::abort();                                \
    }

aclDataType getAclDataType(diopiDtype_t type);
aclDataType getAclDataType(diopiConstTensorHandle_t th);

inline std::string dumpTensor(diopiConstTensorHandle_t th) {
    std::stringstream stream;
    stream << "Tensor(handle:" << th;
    if (th) {
        diopiSize_t shape;
        diopiSize_t stride;
        const void* ptr;
        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);
        diopiGetTensorDataConst(th, &ptr);
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        stream << " ,data:" << ptr;
        stream << " ,dtype:" << dtype;
        stream << " ,shape:";
        std::for_each(shape.data, shape.data + shape.len, [&stream](int64_t v) { stream << v << " "; });
        stream << " ,stride:";
        std::for_each(stride.data, stride.data + stride.len, [&stream](int64_t v) { stream << v << " "; });
    }
    stream << ")";
    return stream.str();
}

inline aclFormat getAclDataFormat(diopiConstTensorHandle_t th) {
    diopiSize_t shape;
    diopiSize_t stride;
    diopiGetTensorShape(th, &shape);
    diopiGetTensorStride(th, &stride);
    check_args(stride.len == shape.len, "stride.len == shape.len check failed");
    if (shape.len == 4) {
        std::array<int64_t, 4> thStride{stride.data[0], stride.data[1], stride.data[2], stride.data[3]};
        {
            std::array<int64_t, 4> nchwStride;
            int st = 1;
            for (auto k : {3, 2, 1, 0}) {
                nchwStride[k] = st;
                if (shape.data[k] == 0) continue;
                if (shape.data[k] == -1) st = -1;
                if (st != -1) st *= shape.data[k];
            }
            if (thStride == nchwStride) {
                return ACL_FORMAT_NCHW;
            }
        }
        std::array<int64_t, 4> nhwcStride;
        int st = 1;
        for (auto k : {1, 3, 2, 0}) {
            nhwcStride[k] = st;
            if (shape.data[k] == 0) continue;
            if (shape.data[k] == -1) st = -1;
            if (st != -1) st *= shape.data[k];
        }
        if (thStride == nhwcStride) {
            return ACL_FORMAT_NHWC;
        }
        warning("Acl only support NCHW or NHWC format! but get %s", dumpTensor(th).c_str());
    }
    return ACL_FORMAT_ND;
}

inline bool is_integral_type(const diopiDtype_t& type) {
    switch (type) {
        case diopi_dtype_bool:
        case diopi_dtype_int8:
        case diopi_dtype_uint8:
        case diopi_dtype_int16:
        case diopi_dtype_uint16:
        case diopi_dtype_int32:
        case diopi_dtype_uint32:
        case diopi_dtype_int64:
        case diopi_dtype_uint64:
            return true;
    }
    return false;
}

template <typename T>
T getValue(const diopiScalar_t* scalar) {
    check_args(scalar != nullptr, "input should not be nullptr");
    if (is_integral_type(scalar->stype)) {
        return static_cast<T>(scalar->ival);
    } else {
        return static_cast<T>(scalar->fval);
    }
}

template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiConstTensorHandle_t) = getAclDataType>
class AclOpRunner {
    std::string opname_;
    aclopAttr* attr_;
    std::array<aclTensorDesc*, InputSize> inputDescs_;
    std::array<aclDataBuffer*, InputSize> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;

    std::string dumpRunnerInfo() {
        std::stringstream sstream;
        sstream << "opname:" << opname_ << ",ins.size:" << InputSize << ",outs.size:" << OutputSize << std::endl;
        return sstream.str();
    }

public:
    explicit AclOpRunner(std::string opname) : opname_(std::move(opname)), attr_(aclopCreateAttr()) {
        inputDescs_.fill(nullptr);
        inputBuffers_.fill(nullptr);
        outputDescs_.fill(nullptr);
        inputBuffers_.fill(nullptr);
    }

    ~AclOpRunner() {
        aclopDestroyAttr(attr_);
        auto destoryAclTensorDesc = [](aclTensorDesc* desc) {
            if (desc) {
                aclDestroyTensorDesc(desc);
            }
        };
        auto destoryAclDataBuffer = [](aclDataBuffer* buffer) {
            if (buffer) {
                aclDestroyDataBuffer(buffer);
            }
        };
        std::for_each(inputDescs_.begin(), inputDescs_.end(), destoryAclTensorDesc);
        std::for_each(outputDescs_.begin(), outputDescs_.end(), destoryAclTensorDesc);
        std::for_each(inputBuffers_.begin(), inputBuffers_.end(), destoryAclDataBuffer);
        std::for_each(outputBuffers_.begin(), outputBuffers_.end(), destoryAclDataBuffer);
    }

    AclOpRunner& addConstInput(const int index, diopiConstTensorHandle_t th, const aclFormat& format) {
        check_args(th != nullptr, "input should not be nullptr");
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        const void* ptr = nullptr;
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        diopiGetTensorNumel(th, &numel);
        diopiGetTensorElemSize(th, &itemsize);
        diopiGetTensorDataConst(th, &ptr);

        std::vector<int64_t> dims(shape.len);
        for (size_t i = 0; i < dims.size(); ++i) {
            dims[i] = shape.data[i];
        }
        if (dims.size() == 0 && numel == 1) {
            dims.push_back(1);
        }

        int finalIndex = index;
        if (index < 0) {
            for (size_t i = 0; i < InputSize; i++) {
                if (inputDescs_[i] == nullptr) {
                    finalIndex = i;
                    break;
                }
            }
        }

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), finalIndex, dumpTensor(th).c_str());
        }

        check_args(finalIndex >= 0 && finalIndex < InputSize, "check 0<=finalIndex<InputSize failed");

        auto& desc = inputDescs_[finalIndex];
        auto& buffer = inputBuffers_[finalIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(th), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        if (numel > 0) CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), numel * itemsize));
        buffer = aclCreateDataBuffer(nullptr, 0);
        return *this;
    }

    template <int index = -1>
    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, const aclFormat& format) {
        static_assert(index < InputSize);
        return addConstInput(index, th, format);
    }

    template <int index = -1>
    AclOpRunner& addConstInput(diopiConstTensorHandle_t th) {
        static_assert(index < InputSize);
        return addConstInput(index, th, getAclDataFormat(th));
    }

    AclOpRunner& addInput(const int index, diopiConstTensorHandle_t th, const aclFormat& format) {
        check_args(th != nullptr, "input should not be nullptr");
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        const void* ptr = nullptr;
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        diopiGetTensorNumel(th, &numel);
        diopiGetTensorElemSize(th, &itemsize);
        diopiGetTensorDataConst(th, &ptr);

        std::vector<int64_t> dims(shape.len);
        for (size_t i = 0; i < dims.size(); ++i) {
            dims[i] = shape.data[i];
        }
        if (dims.size() == 0 && numel == 1) {
            dims.push_back(1);
        }

        int finalIndex = index;
        if (index < 0) {
            for (size_t i = 0; i < InputSize; i++) {
                if (inputDescs_[i] == nullptr) {
                    finalIndex = i;
                    break;
                }
            }
        }

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), finalIndex, dumpTensor(th).c_str());
        }

        check_args(finalIndex >= 0 && finalIndex < InputSize, "check 0<=finalIndex<InputSize failed");

        auto& desc = inputDescs_[finalIndex];
        auto& buffer = inputBuffers_[finalIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(th), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), numel * itemsize);
        return *this;
    }

    template <int index = -1>
    AclOpRunner& addInput(diopiConstTensorHandle_t th, const aclFormat& format) {
        static_assert(index < InputSize);
        return addInput(index, th, format);
    }

    AclOpRunner& addInput(const int index, diopiConstTensorHandle_t th) { return addInput(index, th, getAclDataFormat(th)); }

    template <int index = -1>
    AclOpRunner& addInput(diopiConstTensorHandle_t th) {
        static_assert(index < InputSize);
        return addInput(index, th, getAclDataFormat(th));
    }

    template <typename... Ins>
    AclOpRunner& addInput(diopiConstTensorHandle_t in, const Ins&... ins) {
        addInput<-1>(in).addInput(ins...);
        return *this;
    }

    AclOpRunner& addOutput(const int index, diopiTensorHandle_t th, const aclFormat format) {
        check_args(th != nullptr, "output should not be nullptr");
        int finalIndex = index;
        if (index < 0) {
            for (size_t i = 0; i < OutputSize; i++) {
                if (outputDescs_[i] == nullptr) {
                    finalIndex = i;
                    break;
                }
            }
        }
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s output[%d]:%s", opname_.c_str(), finalIndex, dumpTensor(th).c_str());
        }
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        void* ptr = nullptr;
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        diopiGetTensorNumel(th, &numel);
        diopiGetTensorElemSize(th, &itemsize);
        diopiGetTensorData(th, &ptr);

        std::vector<int64_t> dims(shape.len);
        for (size_t i = 0; i < dims.size(); ++i) {
            dims[i] = shape.data[i];
        }
        if (dims.size() == 0 && numel == 1) {
            dims.push_back(1);
        }

        check_args(finalIndex >= 0 && finalIndex < OutputSize, "check 0<=finalIndex<OutputSize failed");
        auto& desc = outputDescs_[finalIndex];
        auto& buffer = outputBuffers_[finalIndex];
        desc = aclCreateTensorDesc(dtypeCastStrategy(th), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, numel * itemsize);
        return *this;
    }

    template <int index = -1>
    AclOpRunner& addOutput(diopiTensorHandle_t th) {
        return addOutput(index, th, getAclDataFormat(th));
    }

    template <int index = -1>
    AclOpRunner& addOutput(diopiTensorHandle_t th, const aclFormat format) {
        return addOutput(index, th, format);
    }

    template <typename... Outs>
    AclOpRunner& addOutput(diopiTensorHandle_t out, Outs&... outs) {
        return addOutput<-1>(out).addOutput(outs...);
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        if constexpr (std::is_same<T, int64_t>::value || std::is_same<T, int>::value) {
            CALL_ACLRT(aclopSetAttrInt(attr_, attrName.data(), value));
            return *this;
        }
        if constexpr (std::is_same<T, float>::value) {
            CALL_ACLRT(aclopSetAttrFloat(attr_, attrName.data(), value));
            return *this;
        }
        if constexpr (std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value) {
            CALL_ACLRT(aclopSetAttrBool(attr_, attrName.data(), value));
            return *this;
        }
        if constexpr (std::is_same<T, std::string>::value) {
            CALL_ACLRT(aclopSetAttrString(attr_, attrName.data(), value.data()));
            return *this;
        }
        check_args(false, "%s: no specialization for %s type.", dumpRunnerInfo().c_str(), typeid(T).name());
        return *this;
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const typename std::vector<T>& value) {
        std::vector<int64_t> vec(value.begin(), value.end());
        CALL_ACLRT(aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
        return *this;
    }

    template <aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
    AclOpRunner& run(diopiContextHandle_t& ctx) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        int inSize = 0;
        for (size_t i = 0; i < InputSize; i++) {
            if (inputDescs_[i] != nullptr) {
                inSize++;
            }
        }
        int outSize = 0;
        for (size_t i = 0; i < OutputSize; i++) {
            if (outputDescs_[i] != nullptr) {
                outSize++;
            }
        }

        CALL_ACLRT(aclopCompileAndExecute(opname_.data(),
                                          inSize,
                                          inputDescs_.data(),
                                          inputBuffers_.data(),
                                          outSize,
                                          outputDescs_.data(),
                                          outputBuffers_.data(),
                                          attr_,
                                          EngineType,
                                          CompileType,
                                          nullptr,
                                          stream));
        CALL_ACLRT(aclrtSynchronizeStream(stream));
        // check_args(errorcode == ACL_SUCCESS, dumpRunnerInfo().c_str());
        // Get environment variables once when run is called for the first time
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info(dumpRunnerInfo().c_str());
        }

        return *this;
    }
};

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out);

template <typename T>
diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype) {
    int64_t len = size->getLen();
    int64_t buffersize = len * aclDataTypeSize(getAclDataType(dtype));
    int64_t sizeTmp[1] = {len};
    diopiSize_t sSize(sizeTmp, 1);
    diopiRequireTensor(ctx, out, &sSize, nullptr, dtype, diopi_host);
    if (len > 0) {
        void* dst = nullptr;
        diopiGetTensorData(*out, &dst);
        for (int i = 0; i < len; i++) {
            reinterpret_cast<T*>(dst)[i] = (T)size->data[i];
        }
    }
    return diopiSuccess;
}

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
