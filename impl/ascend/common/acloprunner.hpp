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

#define error(...)                               \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");                                \
    std::abort();

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

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out);

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype);

/**
 * @brief some op originally support positive tensor, but ascend op can handle negative tensor. So we need to change those out value to nan
 * @param[in] ctx Context environment.
 * @param[in] input the input tensor.
 * @param[out] the output tensor.
 */
diopiError_t negativeInputRtnFillNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiDtype_t) = getAclDataType>
class AclOpRunner {
    std::string opname_;
    aclopAttr* attr_;
    std::array<aclTensorDesc*, InputSize> inputDescs_;
    std::array<aclDataBuffer*, InputSize> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;
    diopiContextHandle_t context_;
    int inputIndex = 0;
    int outputIndex = 0;

    std::string dumpRunnerInfo() {
        std::stringstream sstream;
        sstream << "opname:" << opname_ << ",ins.size:" << InputSize << ",outs.size:" << OutputSize << std::endl;
        return sstream.str();
    }

public:
    explicit AclOpRunner(std::string opname, diopiContextHandle_t context) : context_(context), opname_(std::move(opname)), attr_(aclopCreateAttr()) {
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

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, const aclFormat& format) {
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

        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);

        std::vector<int64_t> dims(shape.len);
        for (size_t i = 0; i < dims.size(); ++i) {
            dims[i] = shape.data[i];
        }
        if (dims.size() == 0 && numel == 1) {
            dims.push_back(1);
        }

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex, dumpTensor(th).c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);

        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        if (numel > 0) CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), numel * itemsize));
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th) {
        addConstInput(th, getAclDataFormat(th));
        return *this;
    }

    AclOpRunner& addConstInput(diopiTensorHandle_t th) {
        addConstInput(reinterpret_cast<diopiConstTensorHandle_t>(th));
        return *this;
    }

    AclOpRunner& addConstInput(diopiSize_t& size, diopiDtype_t dtype) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor, dtype);
        addConstInput(sizeTensor, ACL_FORMAT_ND);
        return *this;
    }

    AclOpRunner& addConstInput(diopiSize_t& size) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor);
        addConstInput(sizeTensor, ACL_FORMAT_ND);
        return *this;
    }

    template <typename T>
    AclOpRunner& addConstInput(T val) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream ss;
            ss << val;
            info("%s scalar input[%d]: %s", opname_.c_str(), inputIndex, ss.str().c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");
        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];
        diopiDtype_t type;
        if constexpr (std::is_same<T, int64_t>::value) {
            type = diopi_dtype_int64;
        } else if constexpr (std::is_same<T, int32_t>::value) {
            type = diopi_dtype_int32;
        } else if constexpr (std::is_same<T, int16_t>::value) {
            type = diopi_dtype_int16;
        } else if constexpr (std::is_same<T, int8_t>::value) {
            type = diopi_dtype_int8;
        } else if constexpr (std::is_same<T, uint64_t>::value) {
            type = diopi_dtype_uint64;
        } else if constexpr (std::is_same<T, uint32_t>::value) {
            type = diopi_dtype_uint32;
        } else if constexpr (std::is_same<T, uint16_t>::value) {
            type = diopi_dtype_uint16;
        } else if constexpr (std::is_same<T, uint8_t>::value) {
            type = diopi_dtype_uint8;
        } else if constexpr (std::is_same<T, double>::value) {
            type = diopi_dtype_float64;
        } else if constexpr (std::is_same<T, float>::value) {
            type = diopi_dtype_float32;
        } else if constexpr (std::is_same<T, bool>::value) {
            type = diopi_dtype_bool;
        } else {
            error("type not supported: %s", typeid(T).name());
        }
        desc = aclCreateTensorDesc(dtypeCastStrategy(type), 0, nullptr, ACL_FORMAT_ND);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        T valCopy = val;
        CALL_ACLRT(aclSetTensorConst(desc, reinterpret_cast<void*>(&valCopy), 1 * sizeof(T)));
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex++;
        return *this;
    }

    template <typename T>
    AclOpRunner& addConstInput(diopiScalar_t& scalar) {
        T val = getValue<T>(&scalar);
        addConstInput<T>(val);
        return *this;
    }

    AclOpRunner& addConstInput(diopiScalar_t& scalar) {
        if (scalar.stype == diopi_dtype_int64) {
            addConstInput<int64_t>(scalar.ival);
        } else {
            addConstInput<double>(scalar.fval);
        }
        return *this;
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th, const aclFormat& format) {
        check_args(th != nullptr, "input should not be nullptr");
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        const void* ptr = nullptr;
        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);
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

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex, dumpTensor(th).c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), numel * itemsize);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th) { return addInput(th, getAclDataFormat(th)); }

    template <typename... Ins>
    AclOpRunner& addInput(diopiConstTensorHandle_t in, const Ins&... ins) {
        addInput(in).addInput(ins...);
        return *this;
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th, const aclFormat format) {
        check_args(th != nullptr, "output should not be nullptr");

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s output[%d]:%s", opname_.c_str(), outputIndex, dumpTensor(th).c_str());
        }
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        void* ptr = nullptr;
        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);
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

        check_args(outputIndex >= 0 && outputIndex < OutputSize, "check 0<=outputIndex<OutputSize failed");
        auto& desc = outputDescs_[outputIndex];
        auto& buffer = outputBuffers_[outputIndex];
        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, numel * itemsize);
        outputIndex++;
        return *this;
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th) { return addOutput(th, getAclDataFormat(th)); }

    template <typename... Outs>
    AclOpRunner& addOutput(diopiTensorHandle_t out, Outs&... outs) {
        return addOutput(out).addOutput(outs...);
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
    AclOpRunner& run() {
        diopiStreamHandle_t stream;
        diopiGetStream(context_, &stream);

        CALL_ACLRT(aclopCompileAndExecute(opname_.data(),
                                          inputIndex,
                                          inputDescs_.data(),
                                          inputBuffers_.data(),
                                          outputIndex,
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

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
