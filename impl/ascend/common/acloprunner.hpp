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
#include <map>
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

inline std::string dumpTensor(diopiConstTensorHandle_t th, std::string input = "") {
    std::stringstream stream;
    stream << "Tensor(handle:" << th;
    stream << input << std::endl;
    if (th) {
        diopiSize_t shape;
        diopiSize_t stride;
        const void* ptr;
        diopiDtype_t dtype;
        diopiDevice_t device;
        diopiGetTensorDtype(th, &dtype);
        diopiGetTensorDataConst(th, &ptr);
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        diopiGetTensorDevice(th, &device);
        stream << " ,data:" << ptr;
        stream << " ,dtype:" << dtype;
        stream << " ,device:" << device;
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

inline bool isIntegralType(const diopiDtype_t& type) { return type < 8; }

inline bool isIntegralTypeWithBool(const diopiDtype_t& type) { return type < 8 || type == 11; }

inline bool isFloatingType(const diopiDtype_t& type) { return (type <= 10 && type >= 8) || type == 12 || type == 13; }

template <typename T>
T getValue(const diopiScalar_t* scalar) {
    check_args(scalar != nullptr, "input should not be nullptr");
    if (isIntegralTypeWithBool(scalar->stype)) {
        return static_cast<T>(scalar->ival);
    } else {
        return static_cast<T>(scalar->fval);
    }
}

diopiError_t fillTensor(diopiContextHandle_t ctx, diopiTensorHandle_t* out, float val);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, diopiTensorHandle_t* out, diopiDtype_t dtype,
                                  diopiDevice_t device = diopiDevice_t::diopi_host);
diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out);

diopiError_t makeTensorFromSize(diopiContextHandle_t ctx, const diopiSize_t* size, diopiTensorHandle_t* out, diopiDtype_t dtype);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype);

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src);

diopiError_t makeOnesLike(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t src, diopiDtype_t dtype);

diopiTensorHandle_t hostToDevice(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

inline std::vector<int64_t> calcStrides(int ndims, diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    std::vector<int64_t> strides;
    strides.resize(ndims);
    int64_t st = 1;
    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = ndims; i > 0; --i) {
            strides[i - 1] = st;
            if (size.data[i - 1] == 0) continue;
            if (size.data[i - 1] == -1) st = -1;
            if (st != -1) st *= size.data[i - 1];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) st *= size.data[k];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }

    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        for (auto k : {1, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }
    } else {
        // PARROTS_THROW(InvalidArgs) <<
        //         "Invalid MemoryFormat " << memoryFormatName(format);
    }
    return strides;
}

inline bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format = diopiMemoryFormat_t::ChannelsLast) {
    diopiSize_t shape, stride;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    if (shape.len != 4) return false;
    int64_t totalSize = 1;
    for (int64_t i = 0; i < shape.len; ++i) {
        totalSize *= shape.data[i];
    }
    if (totalSize == 0) return false;
    if (stride.data[0] == stride.data[1]) return false;
    if (checkContiguous) {
        auto realStride = calcStrides(shape.len, shape, format);
        for (int i = 0; i < stride.len; ++i) {
            if (i >= realStride.size() || realStride[i] != stride.data[i]) {
                return false;
            }
        }
        return true;
    } else {
        int64_t st = 1;
        std::vector<int> orders;
        if (format == diopiMemoryFormat_t::ChannelsLast)
            orders = {1, 3, 2, 0};
        else if (format == diopiMemoryFormat_t::ChannelsLast3d)
            orders = {1, 4, 3, 2, 0};
        for (auto k : orders) {
            if (stride.data[k] < st) return false;
            st = stride.data[k] * shape.data[k];
        }
        return true;
    }
}

inline diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch = false) {
    return isLikeChannelsLast(tensor, exactMatch)
               ? diopiMemoryFormat_t::ChannelsLast
               : (isLikeChannelsLast(tensor, exactMatch, diopiMemoryFormat_t::ChannelsLast3d) ? diopiMemoryFormat_t::ChannelsLast3d
                                                                                              : diopiMemoryFormat_t::Contiguous);
}

bool isContiguous(diopiConstTensorHandle_t tensor, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

diopiTensorHandle_t clone(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src);

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiDtype_t dtype);

int64_t getBaseBufferSize(diopiConstTensorHandle_t src);

std::vector<int64_t> getBaseShape(diopiConstTensorHandle_t src);

diopiSize_t vectorToDiopiSize(std::vector<int64_t>& sizeVec);

diopiSize_t arrayToDiopiSize(int64_t* data, int64_t len);

template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiDtype_t) = getAclDataType>
class AclOpRunner {
    std::string opname_;
    aclopAttr* attr_;
    std::array<aclTensorDesc*, InputSize> inputDescs_;
    std::array<aclDataBuffer*, InputSize> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;
    std::vector<int64_t> syncIdxs;
    std::vector<diopiTensorHandle_t*> syncTensors;
    std::vector<std::pair<diopiTensorHandle_t, diopiTensorHandle_t>> nonContiguousOutputPairs;
    diopiContextHandle_t context_;
    int inputIndex = 0;
    int outputIndex = 0;
    bool sync = false;

    std::string dumpRunnerInfo() {
        std::stringstream sstream;
        sstream << "opname:" << opname_ << ",ins.size:" << inputIndex << ",outs.size:" << outputIndex << std::endl;
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

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, const aclFormat& format, bool isScalar = false) {
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

        if (isScalar)
            desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), 0, nullptr, format);
        else
            desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);

        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        if (isScalar) {
            CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), itemsize));
        } else {
            if (numel > 0) CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), numel * itemsize));
        }
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addConstInput(const void* ptr, int64_t buffersize, std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype,
                               bool isScalar = false) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s output[%d]: %s", opname_.c_str(), outputIndex, stream.str().c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);

        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), buffersize));
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, bool isScalar = false) {
        addConstInput(th, getAclDataFormat(th), isScalar);
        return *this;
    }

    AclOpRunner& addConstInput(diopiTensorHandle_t th, bool isScalar = false) {
        addConstInput(reinterpret_cast<diopiConstTensorHandle_t>(th), isScalar);
        return *this;
    }

    AclOpRunner& addConstInput(const diopiSize_t& size, diopiDtype_t dtype) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor, dtype);
        addConstInput(sizeTensor, ACL_FORMAT_ND, false);
        return *this;
    }

    AclOpRunner& addConstInput(const diopiSize_t& size) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor);
        addConstInput(sizeTensor, ACL_FORMAT_ND, false);
        return *this;
    }

    AclOpRunner& addConstInput(const diopiScalar_t& scalar, diopiDtype_t dtype) {
        diopiTensorHandle_t scalarTensor;
        makeTensorFromScalar(context_, &scalar, &scalarTensor, dtype);
        addConstInput(scalarTensor, ACL_FORMAT_ND, true);
        return *this;
    }

    AclOpRunner& addConstInput(const diopiScalar_t& scalar) {
        addConstInput(scalar, scalar.stype);
        return *this;
    }

    AclOpRunner& addConstInput(const double val, diopiDtype_t dtype) {
        diopiScalar_t scalar = diopiScalar_t();
        if (isIntegralTypeWithBool(dtype)) {
            scalar.stype = diopi_dtype_int64;
            scalar.ival = static_cast<int64_t>(val);
        } else {
            scalar.stype = diopi_dtype_float64;
            scalar.fval = val;
        }
        addConstInput(scalar, dtype);
        return *this;
    }

    AclOpRunner& addInput(const void* ptr, int64_t buffersize, std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s input[%d]: %s", opname_.c_str(), inputIndex, stream.str().c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), buffersize);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th, const aclFormat& format) {
        check_args(th != nullptr, "input should not be nullptr");
        const void* ptr = nullptr;
        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);
        diopiGetTensorDataConst(th, &ptr);

        std::vector<int64_t> dims = getBaseShape(th);
        int64_t buffSize = getBaseBufferSize(th);

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex, dumpTensor(th).c_str());
        }

        check_args(inputIndex >= 0 && inputIndex < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex];
        auto& buffer = inputBuffers_[inputIndex];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), buffSize);
        inputIndex++;
        return *this;
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th) {
        auto thCopy = contiguous(context_, th);
        return addInput(thCopy, getAclDataFormat(thCopy));
    }

    AclOpRunner& addInputWithoutContiguous(diopiConstTensorHandle_t th) { return addInput(th, getAclDataFormat(th)); }

    AclOpRunner& addInput(diopiConstTensorHandle_t th, diopiDtype_t dtype) {
        auto thCopy = contiguous(context_, th, dtype);
        return addInput(thCopy, getAclDataFormat(thCopy));
    }

    AclOpRunner& addOutput(void* ptr, int64_t buffersize, std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s output[%d]: %s", opname_.c_str(), outputIndex, stream.str().c_str());
        }

        check_args(outputIndex >= 0 && outputIndex < OutputSize, "check 0<=outputIndex<OutputSize failed");
        auto& desc = outputDescs_[outputIndex];
        auto& buffer = outputBuffers_[outputIndex];
        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, buffersize);
        outputIndex++;
        return *this;
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th, const aclFormat format) {
        check_args(th != nullptr, "output should not be nullptr");

        void* ptr = nullptr;
        diopiDtype_t dtype;
        diopiGetTensorDtype(th, &dtype);
        diopiGetTensorData(th, &ptr);

        std::vector<int64_t> dims = getBaseShape(th);
        int64_t buffSize = getBaseBufferSize(th);

        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s output[%d]:%s", opname_.c_str(), outputIndex, dumpTensor(th).c_str());
        }
        check_args(outputIndex >= 0 && outputIndex < OutputSize, "check 0<=outputIndex<OutputSize failed");
        auto& desc = outputDescs_[outputIndex];
        auto& buffer = outputBuffers_[outputIndex];
        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        check_args(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, buffSize);
        outputIndex++;
        return *this;
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th) {
        if (isContiguous(th)) {
            return addOutput(th, getAclDataFormat(th));
        } else {
            diopiTensorHandle_t thCopy;
            diopiSize_t shape;
            diopiDtype_t dtype;
            diopiGetTensorShape(th, &shape);
            diopiGetTensorDtype(th, &dtype);
            diopiRequireTensor(context_, &thCopy, &shape, nullptr, dtype, diopi_device);
            nonContiguousOutputPairs.push_back(std::pair<diopiTensorHandle_t, diopiTensorHandle_t>(th, thCopy));
            addOutput(thCopy, getAclDataFormat(th));
        }
    }

    AclOpRunner& addOutputWithoutContiguous(diopiTensorHandle_t th) { return addOutput(th, getAclDataFormat(th)); }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream stream;
            stream << "attrName=" << attrName;
            stream << ",dtype=" << typeid(T).name();
            stream << ", Attr:(";
            stream << value;
            stream << ")";
            info("%s attr: %s", opname_.c_str(), stream.str().c_str());
        }
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
        if constexpr (std::is_same<T, aclDataType>::value) {
            CALL_ACLRT(aclopSetAttrDataType(attr_, attrName.data(), value));
            return *this;
        }
        check_args(false, "%s: no specialization for %s type.", dumpRunnerInfo().c_str(), typeid(T).name());
        return *this;
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const typename std::vector<T>& value) {
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            std::stringstream stream;
            stream << "attrName=" << attrName;
            stream << ",dtype=" << typeid(T).name();
            stream << ", Attr:(";
            std::for_each(value.data(), value.data() + value.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s attr: %s", opname_.c_str(), stream.str().c_str());
        }
        std::vector<int64_t> vec(value.begin(), value.end());
        CALL_ACLRT(aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
        return *this;
    }

    AclOpRunner& addSyncOutput(diopiTensorHandle_t* th, const aclFormat format) {
        syncIdxs.push_back(outputIndex);
        syncTensors.push_back(th);
        addOutput(*th, format);
        sync = true;
        return *this;
    }

    AclOpRunner& addSyncOutput(diopiTensorHandle_t* th) {
        syncIdxs.push_back(outputIndex);
        syncTensors.push_back(th);
        addOutput(*th, getAclDataFormat(*th));
        sync = true;
        return *this;
    }

    template <aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
    AclOpRunner& run() {
        diopiStreamHandle_t stream;
        diopiGetStream(context_, &stream);
        if (sync) {
            CALL_ACLRT(aclopCompileAndExecuteV2(opname_.data(),
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
        } else {
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
        }
        for (auto pair : nonContiguousOutputPairs) {
            auto th = pair.first;
            auto thCopy = pair.second;
            diopiCopyInp(context_, thCopy, th);
        }
        for (int64_t i = 0; i < syncIdxs.size(); i++) {
            auto syncIdx = syncIdxs[i];
            auto syncTensorPtr = syncTensors[i];
            int descNumDims = aclGetTensorDescNumDims(outputDescs_[syncIdx]);
            std::vector<int64_t> realShape;
            int64_t dimSize = 0;
            for (int64_t j = 0; j < descNumDims; j++) {
                CALL_ACLRT(aclGetTensorDescDimV2(outputDescs_[syncIdx], j, &dimSize));
                realShape.push_back(dimSize);
            }
            diopiTensorHandle_t syncTensorReal;
            diopiSize_t syncTensorRealSize = vectorToDiopiSize(realShape);
            diopiDtype_t dtype;
            diopiGetTensorDtype(*syncTensorPtr, &dtype);
            diopiRequireTensor(context_, &syncTensorReal, &syncTensorRealSize, nullptr, dtype, diopi_device);
            int64_t elemsize, numel, buffersize;
            diopiGetTensorElemSize(syncTensorReal, &elemsize);
            diopiGetTensorNumel(syncTensorReal, &numel);
            void *dst, *src;
            diopiGetTensorData(*syncTensorPtr, &src);
            diopiGetTensorData(syncTensorReal, &dst);
            buffersize = numel * elemsize;
            if (buffersize > 0 && src != nullptr && dst != nullptr) {
                CALL_ACLRT(aclrtMemcpyAsync(dst, buffersize, src, buffersize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            }
            *syncTensorPtr = syncTensorReal;
        }
        CALL_ACLRT(aclrtSynchronizeStream(stream));
        // Get environment variables once when run is called for the first time
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info(dumpRunnerInfo().c_str());
        }

        return *this;
    }
};

static inline diopiDtype_t promoteTypes(diopiDtype_t a, diopiDtype_t b) {
    // This is generated according to NumPy's promote_types
    constexpr auto u1 = diopi_dtype_uint8;
    constexpr auto i1 = diopi_dtype_int8;
    constexpr auto i2 = diopi_dtype_int16;
    constexpr auto i4 = diopi_dtype_int32;
    constexpr auto i8 = diopi_dtype_int64;
    constexpr auto f2 = diopi_dtype_float16;
    constexpr auto f4 = diopi_dtype_float32;
    constexpr auto f8 = diopi_dtype_float64;
    constexpr auto c4 = diopi_dtype_complex64;
    constexpr auto c8 = diopi_dtype_complex128;
    constexpr auto b1 = diopi_dtype_bool;

    static std::map<diopiDtype_t, int> dtypeMap = {{u1, 0}, {i1, 1}, {i2, 2}, {i4, 3}, {i8, 4}, {f2, 5}, {f4, 6}, {f8, 7}, {c4, 8}, {c8, 9}, {b1, 10}};
    static constexpr diopiDtype_t promoteTypesLookup[11][11] = {
        /*        u1  i1  i2  i4  i8  f2  f4  f8  c4  c8  b1*/
        /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c4, c8, u1},
        /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c4, c8, i1},
        /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c4, c8, i2},
        /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c4, c8, i4},
        /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c4, c8, i8},
        /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c4, c8, f2},
        /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, f4},
        /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, f8},
        /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c8, c4},
        /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
        /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c4, c8, b1},
    };

    check_args((dtypeMap.count(a) != 0 && dtypeMap.count(b) != 0), "dtype a %d or b %d not supported.", a, b);
    return promoteTypesLookup[dtypeMap[a]][dtypeMap[b]];
}

std::pair<uint64_t, int64_t> getSeedAndOffset(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, uint64_t inc);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
