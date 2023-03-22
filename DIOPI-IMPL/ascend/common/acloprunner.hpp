#ifndef ACLOPRUNNER_HPP_
#define ACLOPRUNNER_HPP_

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <diopi/diopirt.h>

#include <array>
#include <sstream>
#include <vector>

namespace impl {
namespace ascend {

#define CALL_ACLRT(Expr)                                                                \
    {                                                                                   \
        ::aclError ret = Expr;                                                          \
        if (ret != ::ACL_SUCCESS) {                                                     \
            printf("call a ascendrt function (%s) failed. return code=%d", #Expr, ret); \
        }                                                                               \
    }

#define warning(...)                         \
    printf("[%s:%d]: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__);                     \
    printf("\n");

#define info(...)                            \
    printf("[%s:%d]: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__);                     \
    printf("\n");

#define check_args(condition, ...)               \
    if (!(condition)) {                          \
        printf("[%s:%d]: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                     \
        printf("\n");                            \
        std::abort();                            \
    }

inline aclDataType getAclDataType(const diopiTensorHandle_t& th) {
    diopiDtype_t type;
    diopiGetTensorDtype(th, &type);
    switch (type) {
        case diopi_dtype_float16:
            return ACL_FLOAT16;
        case diopi_dtype_float32:
            return ACL_FLOAT;
        case diopi_dtype_float64:
            return ACL_DOUBLE;
        case diopi_dtype_int8:
            return ACL_INT8;
        case diopi_dtype_uint8:
            return ACL_UINT8;
        case diopi_dtype_int16:
            return ACL_INT16;
        case diopi_dtype_uint16:
            return ACL_UINT16;
        case diopi_dtype_int32:
            return ACL_INT32;
        case diopi_dtype_uint32:
            return ACL_UINT32;
        case diopi_dtype_int64:
            return ACL_INT64;
        case diopi_dtype_uint64:
            return ACL_UINT64;
        case diopi_dtype_bool:
            return ACL_BOOL;
    }
    check_args(false, "acl not support dioptDtype_t:%d", type);
    return ACL_DT_UNDEFINED;
}

inline aclFormat getAclDataFormat(const diopiTensorHandle_t& th) {
    diopiSize_t shape;
    diopiSize_t stride;
    diopiGetTensorShape(th, &shape);
    diopiGetTensorStride(th, &stride);
    if (shape.len == 4) {
        return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}

template <int InputSize, int OutputSize, aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
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
    AclOpRunner(std::string opname) : opname_(std::move(opname)), attr_(aclopCreateAttr()) {
        for (size_t i = 0; i < InputSize; i++) {
            inputDescs_[i] = nullptr;
            inputBuffers_ = nullptr;
        }
        for (size_t i = 0; i < OutputSize; i++) {
            outputDescs_ = nullptr;
            outputBuffers_ = nullptr;
        }
    }

    ~AclOpRunner() {
        aclopDestroyAttr(attr_);
        for (size_t i = 0; i < InputSize; i++) {
            auto desc = inputDescs_[i];
            if (desc) {
                aclDestroyTensorDesc(desc);
            }
            auto buffer = inputDescs_[i];
            if (buffer) {
                aclDestroyDataBuffer(buffer);
            }
        }
        for (size_t i = 0; i < OutputSize; i++) {
            auto desc = outputDescs_[i];
            if (desc) {
                aclDestroyTensorDesc(desc);
            }
            auto buffer = outputDescs_[i];
            if (buffer) {
                aclDestroyDataBuffer(buffer);
            }
        }
    }

    template <int index>
    AclOpRunner& addInput(const diopiTensorHandle_t& th, const aclFormat& format) {
        static_assert(index >= 0 && index < InputSize);
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

        auto& desc = inputDescs_[index];
        auto& buffer = inputBuffers_[index];
        desc = aclCreateTensorDesc(getAclDataType(th), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), numel * itemsize);
    }

    template <int index>
    AclOpRunner& addInput(const diopiTensorHandle_t& th) {
        return addInput<index>(th, getAclDataFormat(th));
    }

    template <int index>
    AclOpRunner& addOutput(diopiTensorHandle_t& th, const aclFormat& format) {
        static_assert(index >= 0 && index < OutputSize);
        diopiSize_t shape;
        diopiSize_t stride;
        int64_t numel = 0;
        int64_t itemsize = 0;
        void* ptr = nullptr;
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

        auto& desc = outputDescs_[index];
        auto& buffer = outputBuffers_[index];
        desc = aclCreateTensorDesc(getAclDataType(th), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, numel * itemsize);
    }

    template <int index>
    AclOpRunner& addOutput(diopiTensorHandle_t& th) {
        return addOutput<index>(th, getAclDataFormat(th));
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        if (std::is_same<T, int64_t>::value || std::is_same<T, int>::value) {
            CALL_ACLRT(aclopSetAttrInt(attr_, attrName.data(), value));
            return *this;
        }
        if (std::is_same<T, float>::value) {
            CALL_ACLRT(aclopSetAttrFloat(attr_, attrName.data(), value));
            return *this;
        }
        if (std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value) {
            CALL_ACLRT(aclopSetAttrBool(attr_, attrName.data(), value));
            return *this;
        }
        if (std::is_same<T, char*>::value) {
            CALL_ACLRT(aclopSetAttrFloat(attr_, attrName.data(), value));
            return *this;
        }

        check_args(false, "no specialization for this type.");
    }

    AclOpRunner& run(diopiContextHandle_t& ctx) {
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);

        auto errorcode = aclopCompileAndExecute(opname_.data(),
                                                InputSize,
                                                inputDescs_.data(),
                                                inputBuffers_.data(),
                                                OutputSize,
                                                outputDescs_.data(),
                                                outputBuffers_.data(),
                                                attr_,
                                                EngineType,
                                                CompileType,
                                                nullptr,
                                                stream);

        check_args(errorcode == ACL_SUCCESS, dumpRunnerInfo().c_str());

        //  Get environment variables once when run is called for the first time
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info(dumpRunnerInfo().c_str());
        }

        return *this;
    }
};

}  // namespace ascend
}  // namespace impl

#endif  //  ACLOPRUNNER_HPP_