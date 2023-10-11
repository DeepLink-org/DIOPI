#ifndef IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
#define IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_

#include <stdint.h>

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "../ascend_tensor.hpp"
#include "acl/acl.h"
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "debug.hpp"
#include "impl_functions.hpp"
#include "utils.hpp"

namespace impl {
namespace ascend {

aclDataType getAclDataType(diopiDtype_t type);
aclDataType getAclDataType(diopiConstTensorHandle_t th);

inline aclFormat getAclDataFormat(diopiConstTensorHandle_t th) {
    diopiSize_t shape;
    diopiSize_t stride;
    diopiGetTensorShape(th, &shape);
    diopiGetTensorStride(th, &stride);
    ASCEND_CHECK_ABORT(stride.len == shape.len, "stride.len == shape.len check failed");
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
    ASCEND_CHECK_ABORT(scalar != nullptr, "input should not be nullptr");
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

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

diopiTensorHandle_t contiguous(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiDtype_t dtype,
                               diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

int64_t getBaseBufferSize(diopiConstTensorHandle_t src);

std::vector<int64_t> getBaseShape(diopiConstTensorHandle_t src);

diopiSize_t vectorToDiopiSize(std::vector<int64_t>& sizeVec);

diopiSize_t arrayToDiopiSize(int64_t* data, int64_t len);

template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiDtype_t) = getAclDataType>
class AclOpRunner {
    std::string opname_;
    aclopAttr* attr_;
    std::vector<aclTensorDesc*> inputDescs_;
    std::vector<aclDataBuffer*> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;
    std::vector<int64_t> syncIdxs_;
    std::vector<diopiTensorHandle_t*> syncTensors_;
    std::vector<std::pair<diopiTensorHandle_t, diopiTensorHandle_t>> nonContiguousOutputPairs_;
    diopiContextHandle_t context_;
    int inputIndex_ = 0;
    int outputIndex_ = 0;
    bool sync_ = false;
    bool hasDynamicInput_ = false;
    int dynamcInputSize_ = -1;

    std::string dumpRunnerInfo() {
        std::stringstream sstream;
        sstream << "opname:" << opname_ << ",ins.size:" << inputIndex_ << ",outs.size:" << outputIndex_ << std::endl;
        return sstream.str();
    }

public:
    explicit AclOpRunner(std::string opname, diopiContextHandle_t context) : opname_(std::move(opname)), attr_(aclopCreateAttr()), context_(context) {
        inputDescs_.resize(InputSize, nullptr);
        inputBuffers_.resize(InputSize, nullptr);
        outputDescs_.fill(nullptr);
        outputBuffers_.fill(nullptr);
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

    /**
     * @brief Retrieve the actual count of input parameters. In the case of dynamic inputs, it returns the number of dynamic tensors.
     * @return the actual count of input parameters.
     */
    int64_t inputSize() { return hasDynamicInput_ ? dynamcInputSize_ : InputSize; }

    AclOpRunner& addConstInput(const AscendTensor& at, const aclFormat& format, bool isScalar = false) {
        ASCEND_CHECK_ABORT(at.defined(), "input should not be nullptr");
        std::vector<int64_t> dims = at.shape();
        if (dims.empty() && at.numel() == 1) {
            dims.push_back(1);
        }

        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex_, dumpTensor(at).c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        if (isScalar) {
            desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), 0, nullptr, format);

        } else {
            desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), dims.size(), dims.data(), format);
        }

        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        if (isScalar) {
            CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(at.data()), at.elemsize()));
        } else {
            if (at.numel() > 0) CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(at.data()), at.numel() * at.elemsize()));
        }
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex_++;
        return *this;
    }

    AclOpRunner& addConstInput(const void* ptr, int64_t buffersize, std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype,
                               bool isScalar = false) {
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s output[%d]: %s", opname_.c_str(), outputIndex_, stream.str().c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < InputSize, "check 0<=inputIndex_<InputSize failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);

        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(ptr), buffersize));
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex_++;
        return *this;
    }

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, const aclFormat& format, bool isScalar = false) {
        AscendTensor at = AscendTensor(th);
        return addConstInput(at, format, isScalar);
    }

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, bool isScalar = false) { return addConstInput(th, getAclDataFormat(th), isScalar); }

    AclOpRunner& addConstInput(diopiTensorHandle_t th, bool isScalar = false) {
        return addConstInput(reinterpret_cast<diopiConstTensorHandle_t>(th), isScalar);
    }

    AclOpRunner& addConstInput(const diopiSize_t& size, diopiDtype_t dtype) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor, dtype);
        return addConstInput(sizeTensor, ACL_FORMAT_ND, false);
    }

    AclOpRunner& addConstInput(const diopiSize_t& size) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor);
        return addConstInput(sizeTensor, ACL_FORMAT_ND, false);
    }

    AclOpRunner& addConstInput(const std::vector<int64_t>& size) { return addConstInput({size.data(), static_cast<int64_t>(size.size())}); }

    AclOpRunner& addConstInput(const diopiScalar_t& scalar, diopiDtype_t dtype) {
        diopiTensorHandle_t scalarTensor;
        makeTensorFromScalar(context_, &scalar, &scalarTensor, dtype);
        return addConstInput(scalarTensor, ACL_FORMAT_ND, true);
    }

    AclOpRunner& addConstInput(const diopiScalar_t& scalar) { return addConstInput(scalar, scalar.stype); }

    AclOpRunner& addConstInput(const double val, diopiDtype_t dtype) {
        diopiScalar_t scalar = diopiScalar_t();
        if (isIntegralTypeWithBool(dtype)) {
            scalar.stype = diopi_dtype_int64;
            scalar.ival = static_cast<int64_t>(val);
        } else {
            scalar.stype = diopi_dtype_float64;
            scalar.fval = val;
        }
        return addConstInput(scalar, dtype);
    }

    AclOpRunner& addInput(const void* ptr, int64_t buffersize, const std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype) {
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s input[%d]: %s", opname_.c_str(), inputIndex_, stream.str().c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < InputSize, "check 0<=inputIndex<InputSize failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), buffersize);
        inputIndex_++;
        return *this;
    }

    AclOpRunner& addInput(const AscendTensor& at, const aclFormat& format) {
        ASCEND_CHECK_ABORT(at.defined(), "input should not be nullptr");

        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex_, dumpTensor(at).c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < inputSize(), "check 0<=inputIndex<inputSize() failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        std::vector<int64_t> dims = at.getAclMemShape();
        desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(at.data()), at.getAclMemBufferSize());
        inputIndex_++;
        return *this;
    }

    AclOpRunner& addInput(const AscendTensor& at) {
        if (at.isContiguous()) {
            return addInput(at, at.getAclDataFormat());
        } else {
            AscendTensor atCopy;
            makeTensorLike(context_, atCopy, at);
            contiguous(context_, at, atCopy);
            return addInput(atCopy, atCopy.getAclDataFormat());
        }
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th, const aclFormat& format) {
        AscendTensor at = AscendTensor(th);
        return addInput(at, format);
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

    /**
     * @brief: support input dynamic length tensors, need only one input(dynamic)
     * @param[in] tensors diopi const tensors pointer.
     * @return A reference to the modified AclOpRunner
     */
    template <typename T>
    AclOpRunner& addDynamicInput(const std::vector<T>& tensors) {
        ASCEND_CHECK_ABORT(hasDynamicInput_ || inputIndex_ == 0 || InputSize == 1, "only support one dynamic input");
        hasDynamicInput_ = true;
        dynamcInputSize_ = tensors.size();
        inputDescs_.resize(dynamcInputSize_);
        inputBuffers_.resize(dynamcInputSize_);
        for (int i = 0; i < dynamcInputSize_; ++i) {
            addInput(tensors[i]);
        }
        return *this;
    }

    AclOpRunner& addOutput(void* ptr, int64_t buffersize, const std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype) {
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            std::stringstream stream;
            stream << "Tensor:(";
            stream << " data:" << ptr;
            stream << " ,dtype:" << dtype;
            stream << " ,shape:";
            std::for_each(dims.data(), dims.data() + dims.size(), [&stream](int64_t v) { stream << v << " "; });
            stream << ")";
            info("%s output[%d]: %s", opname_.c_str(), outputIndex_, stream.str().c_str());
        }

        ASCEND_CHECK_ABORT(outputIndex_ >= 0 && outputIndex_ < OutputSize, "check 0<=outputIndex<OutputSize failed");
        auto& desc = outputDescs_[outputIndex_];
        auto& buffer = outputBuffers_[outputIndex_];
        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, buffersize);
        outputIndex_++;
        return *this;
    }

    AclOpRunner& addOutput(const AscendTensor& at, const aclFormat format) {
        ASCEND_CHECK_ABORT(at.defined(), "output should not be nullptr");

        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            info("%s output[%d]:%s", opname_.c_str(), outputIndex_, dumpTensor(at).c_str());
        }
        ASCEND_CHECK_ABORT(outputIndex_ >= 0 && outputIndex_ < OutputSize, "check 0<=outputIndex<OutputSize failed");
        auto& desc = outputDescs_[outputIndex_];
        auto& buffer = outputBuffers_[outputIndex_];

        std::vector<int64_t> dims = at.getAclMemShape();
        desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        // change const void* to void*
        buffer = aclCreateDataBuffer(const_cast<void*>(at.data()), at.getAclMemBufferSize());
        outputIndex_++;
        return *this;
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th, const aclFormat format) {
        AscendTensor at = AscendTensor(th);
        return addOutput(at, format);
    }

    AclOpRunner& addOutput(const AscendTensor& at) {
        if (at.isContiguous()) {
            return addOutput(at, at.getAclDataFormat());
        } else {
            diopiTensorHandle_t thCopy;
            diopiSize_t shape;
            diopiGetTensorShape(at.tensorHandle(), &shape);
            diopiRequireTensor(context_, &thCopy, &shape, nullptr, at.dtype(), diopi_device);
            nonContiguousOutputPairs_.emplace_back(const_cast<diopiTensorHandle_t>(at.tensorHandle()), thCopy);
            return addOutput(thCopy, at.getAclDataFormat());
        }
    }

    AclOpRunner& addOutput(diopiTensorHandle_t th) {
        AscendTensor at = AscendTensor(th);
        return addOutput(at);
    }

    AclOpRunner& addOutputWithoutContiguous(diopiTensorHandle_t th) { return addOutput(th, getAclDataFormat(th)); }

    template <typename T, std::enable_if_t<std::is_same<T, int64_t>::value || std::is_same<T, int>::value, void*> = nullptr>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        CALL_ACLRT(aclopSetAttrInt(attr_, attrName.data(), value));
        return *this;
    }

    // float, double, long double
    template <typename T, std::enable_if_t<std::is_floating_point<T>::value, void*> = nullptr>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        CALL_ACLRT(aclopSetAttrFloat(attr_, attrName.data(), value));
        return *this;
    }

    template <typename T, std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, bool>::value, void*> = nullptr>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        CALL_ACLRT(aclopSetAttrBool(attr_, attrName.data(), value));
        return *this;
    }

    template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, void*> = nullptr>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        CALL_ACLRT(aclopSetAttrString(attr_, attrName.data(), value.data()));
        return *this;
    }

    template <typename T, std::enable_if_t<std::is_same<T, aclDataType>::value, void*> = nullptr>
    AclOpRunner& setAttr(const std::string& attrName, const T& value) {
        CALL_ACLRT(aclopSetAttrDataType(attr_, attrName.data(), value));
        return *this;
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, ...) {
        ASCEND_CHECK_ABORT(false, "%s: no specialization for %s type.", dumpRunnerInfo().c_str(), typeid(T).name());
        return *this;
    }

    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const typename std::vector<T>& value) {
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
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
        syncIdxs_.push_back(outputIndex_);
        syncTensors_.push_back(th);
        sync_ = true;
        return addOutput(*th, format);
    }

    AclOpRunner& addSyncOutput(diopiTensorHandle_t* th) {
        syncIdxs_.push_back(outputIndex_);
        syncTensors_.push_back(th);
        sync_ = true;
        return addOutput(*th, getAclDataFormat(*th));
    }

    template <aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
    AclOpRunner& run() {
        diopiStreamHandle_t stream;
        diopiGetStream(context_, &stream);
        if (sync_) {
            CALL_ACLRT(aclopCompileAndExecuteV2(opname_.data(),
                                                inputIndex_,
                                                inputDescs_.data(),
                                                inputBuffers_.data(),
                                                outputIndex_,
                                                outputDescs_.data(),
                                                outputBuffers_.data(),
                                                attr_,
                                                EngineType,
                                                CompileType,
                                                nullptr,
                                                stream));
        } else {
            CALL_ACLRT(aclopCompileAndExecute(opname_.data(),
                                              inputIndex_,
                                              inputDescs_.data(),
                                              inputBuffers_.data(),
                                              outputIndex_,
                                              outputDescs_.data(),
                                              outputBuffers_.data(),
                                              attr_,
                                              EngineType,
                                              CompileType,
                                              nullptr,
                                              stream));
        }
        for (auto pair : nonContiguousOutputPairs_) {
            auto th = pair.first;
            auto thCopy = pair.second;
            diopiCopyInp(context_, thCopy, th);
        }
        for (int64_t i = 0; i < syncIdxs_.size(); i++) {
            auto syncIdx = syncIdxs_[i];
            auto syncTensorPtr = syncTensors_[i];
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
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (aclDebugFlag > 0) {
            info("%s", dumpRunnerInfo().c_str());
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

    ASCEND_CHECK_ABORT((dtypeMap.count(a) != 0 && dtypeMap.count(b) != 0), "dtype a %d or b %d not supported.", a, b);
    return promoteTypesLookup[dtypeMap[a]][dtypeMap[b]];
}

std::pair<uint64_t, int64_t> getSeedAndOffset(diopiContextHandle_t ctx, diopiGeneratorHandle_t gen, uint64_t inc);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
