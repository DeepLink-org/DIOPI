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

#include "../ascend_tensor.hpp"
#include "../error.hpp"
#include "acl_adaptor.hpp"
#include "debug.hpp"
#include "tensor_utils.hpp"

namespace impl {
namespace ascend {

/**
 * @brief: This class is an adapter for Ascend Operators.
 * @note Example(use AscendTensor or diopiTensorHandle_t as tensor, suggest use AscendTensor because AscendTensor has even more convenient methods):
 * ```
 * AscendTensor input1;
 * diopiTensorHandle_t input2, output;
 * float val;
 * AclOpRunner<2, 1>("xxx", ctx)
 *     .addInput(input1, {diopi_dtype_int8, diopi_dtype_uint8})
 *     .addInput(input2)
 *     .setAttr("attr1", val)
 *     .addOutput(output)
 *     .run();
 * ```
 *
 * @note Example(support dynamic size input):
 * ```
 * AscendTensor t1, t2, output;
 * std::vector<AscendTensor> vec{t1, t2};
 * AclOpRunner<1, 1>("xxx", ctx)
 *     .addDynamicInput(vec)
 *     .addOutput(output)
 *     .run();
 * ```
 */
template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiDtype_t) = getAclDataType>
class AclOpRunner final {
private:
    std::string opname_;
    aclopAttr* attr_;
    std::vector<aclTensorDesc*> inputDescs_;
    std::vector<aclDataBuffer*> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;
    std::vector<int64_t> syncIdxs;
    std::vector<diopiTensorHandle_t*> syncTensors;
    std::vector<std::pair<diopiTensorHandle_t, diopiTensorHandle_t>> nonContiguousOutputPairs;
    diopiContextHandle_t context_;
    int inputIndex_ = 0;
    int outputIndex_ = 0;
    bool sync_ = false;
    bool hasDynamicInput_ = false;
    int64_t dynamcInputSize_ = -1;

    std::string dumpRunnerInfo() {
        std::stringstream sstream;
        sstream << "opname:" << opname_ << ",ins.size:" << inputIndex_ << ",outs.size:" << outputIndex_ << std::endl;
        return sstream.str();
    }

public:
    explicit AclOpRunner(std::string opname, diopiContextHandle_t context) : context_(context), opname_(std::move(opname)), attr_(aclopCreateAttr()) {
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

    int64_t inputSize() { return hasDynamicInput_ ? dynamcInputSize_ : InputSize; }

    /**
     * @brief: add diopi const tensor to AclOpRuner.
     * @param[in] at ascend tensor
     * @param[in] format tensor format for ascend
     * @param[in] isScalar true if it is a scalar, false otherwise (default is false)
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(AscendTensor& at, const aclFormat& format, bool isScalar = false) {
        ASCEND_CHECK_ABORT(at.defined(), "input should not be nullptr");
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex_, dumpTensor(at).c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < inputSize(), "check 0<=inputIndex_<inputSize() failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        if (isScalar) {
            desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), 0, nullptr, format);
        } else {
            desc = aclCreateTensorDesc(dtypeCastStrategy(at.dtype()), at.shape().size(), at.shape().data(), format);
        }
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");

        if (isScalar) {
            CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(at.data()), at.elemsize()));
        } else {
            if (at.numel() > 0) CALL_ACLRT(aclSetTensorConst(desc, const_cast<void*>(at.data()), at.getBaseBufferSize()));
        }
        buffer = aclCreateDataBuffer(nullptr, 0);
        inputIndex_++;
        return *this;
    }

    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, const aclFormat& format, bool isScalar = false) {
        AscendTensor at = AscendTensor(th);
        return addConstInput(at, format, isScalar);
    }

    /**
     * @brief: add diopi const tensor to AclOpRuner.
     * @param[in] th const tensor pointer
     * @param[in] isScalar true if it is a scalar, false otherwise (default is false)
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(diopiConstTensorHandle_t th, bool isScalar = false) {
        addConstInput(th, getAclDataFormat(th), isScalar);
        return *this;
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] th tensor pointer
     * @param[in] isScalar true if it is a scalar, false otherwise (default is false)
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(diopiTensorHandle_t th, bool isScalar = false) {
        addConstInput(reinterpret_cast<diopiConstTensorHandle_t>(th), isScalar);
        return *this;
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] size tensor size
     * @param[in] dtype tensor dtype
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(const diopiSize_t& size, diopiDtype_t dtype) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor, dtype);
        return addConstInput(sizeTensor, ACL_FORMAT_ND, false);
    }

    /**
     * @brief: add diopi tensor(construct by size) to AclOpRuner.
     * @param[in] size tensor size
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(const diopiSize_t& size) {
        diopiTensorHandle_t sizeTensor;
        makeTensorFromSize(context_, &size, &sizeTensor);
        return addConstInput(sizeTensor, ACL_FORMAT_ND, false);
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] scalar scalar object
     * @param[in] dtype tensor dtype
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(const diopiScalar_t& scalar, diopiDtype_t dtype) {
        diopiTensorHandle_t scalarTensor;
        makeTensorFromScalar(context_, &scalar, &scalarTensor, dtype);
        return addConstInput(scalarTensor, ACL_FORMAT_ND, true);
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] scalar scalar object
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addConstInput(const diopiScalar_t& scalar) {
        return addConstInput(scalar, scalar.stype);
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] val data
     * @param[in] dtype tensor dtype
     * @return A reference to the modified AclOpRunner
     */
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

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] ptr data pointer
     * @param[in] buffersize data length
     * @param[in] dims target tensor dims
     * @param[in] format target tensor format for ascend
     * @param[in] dtype target tensor dtype
     * @return A reference to the modified AclOpRunner
     */
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
            info("%s input[%d]: %s", opname_.c_str(), inputIndex_, stream.str().c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < inputSize(), "check 0<=inputIndex_<inputSize() failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(ptr), buffersize);
        inputIndex_++;
        return *this;
    }

    /**
     * @brief add ascend tensor with target dtype.
     * @param[in] tensor ascend tensor.
     * @param[in] supportDtypes tensor dtype.
     * @return A reference to the modified AclOpRunner.
     */
    AclOpRunner& addInput(AscendTensor& tensor, const std::set<diopiDtype_t>& supportDtypes) {
        if (supportDtypes.count(tensor.dtype())) {
            return addInput(tensor);
        }

        diopiDtype_t dtype{diopi_dtype_unsupported};
        for (auto item : supportDtypes) {
            if (supportDtypes.count(promoteTypes(item, tensor.dtype()))) {
                dtype = promoteTypes(item, tensor.dtype());
                break;
            }
        }
        ASCEND_CHECK_ABORT(diopi_dtype_unsupported != dtype, "addInput tensor type unsupport dioptDtype_t:%d", dtype);

        diopiSize_t sizeTmp{tensor.shape().data(), tensor.dim()};
        tensor = createAscendTensor(context_, &sizeTmp, nullptr, dtype, tensor.device());

        return addInput(tensor);
    }

    /**
     * @brief: add ascend tensor to AclOpRuner. This function will not change input to contguous.
     * @param[in] th const ascend tensor reference.
     * @param[in] format target tensor format for ascend
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addInput(const AscendTensor& th, const aclFormat& format) {
        ASCEND_CHECK_ABORT(th.defined(), "input should not be nullptr");
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s input[%d]:%s", opname_.c_str(), inputIndex_, dumpTensor(th).c_str());
        }

        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < inputSize(), "check 0<=inputIndex_<inputSize() failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        std::vector<int64_t> dims = th.getBaseShape();
        desc = aclCreateTensorDesc(dtypeCastStrategy(th.dtype()), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(th.data()), th.getBaseBufferSize());
        inputIndex_++;
        return *this;
    }

    /**
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] th diopi const tensor pointer
     * @param[in] format target tensor format for ascend
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addInput(diopiConstTensorHandle_t th, const aclFormat& format) { return addInput(AscendTensor(th), format); }

    /**
     * @brief: add diopi tensor to AclOpRuner. If input tensor without contiguous, change data condiguous.
     * @param[in] at diopi const tensor pointer
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addInput(const AscendTensor& at) {
        AscendTensor& nonConstTensor = const_cast<AscendTensor&>(at);
        contiguous(context_, nonConstTensor);
        return addInput(nonConstTensor, nonConstTensor.getAclDataFormat());
    }

    AclOpRunner& addInput(diopiConstTensorHandle_t th) { return addInput(AscendTensor(th)); }

    /**
     * @brief: add diopi tensor to AclOpRuner. Do not care data stride.
     * @param[in] th diopi const tensor pointer
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addInputWithoutContiguous(diopiConstTensorHandle_t th) { return addInput(th, getAclDataFormat(th)); }

    /**
     * @brief: add diopi tensor to AclOpRuner. If input tensor without contiguous, change data condiguous.
     * @param[in] th diopi const tensor pointer.
     * @param[in] dtype change tensor dtype.
     * @return A reference to the modified AclOpRunner.
     */
    AclOpRunner& addInput(diopiConstTensorHandle_t th, const std::set<diopiDtype_t>& dtype) {
        AscendTensor at = AscendTensor(th);
        return addInput(at, dtype);
    }

    /**
     * @brief: support input dynamic length tensors, need only one input(dynamic)
     * @param[in] tensors diopi const tensors pointer.
     * @return A reference to the modified AclOpRunner
     */
    template <typename T>
    AclOpRunner& addDynamicInput(std::vector<T>& tensors) {
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

    /**
     * @brief: add output diopi tensor to AclOpRuner.
     * @param[out] ptr data pointer
     * @param[in] buffersize data length
     * @param[in] dims target tensor dims
     * @param[in] format target tensor format for ascend
     * @param[in] dtype target tensor dtype
     * @return A reference to the modified AclOpRunner
     */
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
            info("%s output[%d]: %s", opname_.c_str(), outputIndex_, stream.str().c_str());
        }

        ASCEND_CHECK_ABORT(outputIndex_ >= 0 && outputIndex_ < OutputSize, "check 0<=outputIndex_<OutputSize failed");
        auto& desc = outputDescs_[outputIndex_];
        auto& buffer = outputBuffers_[outputIndex_];
        desc = aclCreateTensorDesc(dtypeCastStrategy(dtype), dims.size(), dims.data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(ptr, buffersize);
        outputIndex_++;
        return *this;
    }

    /**
     * @brief add operator output tensor.
     * @param[out] at ascend tensor.
     * @param[in] format tensor format.
     */
    AclOpRunner& addOutput(AscendTensor&& th, const aclFormat format) {
        ASCEND_CHECK_ABORT(th.defined(), "output should not be nullptr");
        static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
            info("%s output[%d]:%s", opname_.c_str(), outputIndex_, dumpTensor(th).c_str());
        }

        ASCEND_CHECK_ABORT(outputIndex_ >= 0 && outputIndex_ < OutputSize, "check 0<=outputIndex_<OutputSize failed");
        auto& desc = outputDescs_[outputIndex_];
        auto& buffer = outputBuffers_[outputIndex_];

        desc = aclCreateTensorDesc(dtypeCastStrategy(th.dtype()), th.getBaseShape().size(), th.getBaseShape().data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(th.data(), th.getBaseBufferSize());
        outputIndex_++;
        return *this;
    }

    AclOpRunner& addOutput(AscendTensor& at, const aclFormat format) { return addOutput(std::move(at), format); }
    AclOpRunner& addOutput(diopiTensorHandle_t th, const aclFormat format) { return addOutput(std::move(AscendTensor(th)), format); }

    /**
     * @brief: add output diopi tensor to AclOpRuner. If output data without contiuous, first change output contiuous,
     *      then call ascend op, finally fill the result data to the ouput tensor.
     * @param[out] at output tensor handle.
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addOutput(AscendTensor&& at) {
        if (at.isContiguous()) {
            return addOutput(std::move(at), at.getAclDataFormat());
        }

        AscendTensor tensor = createAscendTensor(context_, at.shape(), nullptr, at.dtype(), diopi_device);
        // TODO: optimize
        nonContiguousOutputPairs.push_back(std::make_pair(static_cast<diopiTensorHandle_t>(at), static_cast<diopiTensorHandle_t>(tensor)));
        return addOutput(tensor, at.getAclDataFormat());
    }

    AclOpRunner& addOutput(AscendTensor& at) { return addOutput(std::move(at)); }

    AclOpRunner& addOutput(diopiTensorHandle_t th) { return addOutput(std::move(AscendTensor(th))); }

    AclOpRunner& addOutputWithoutContiguous(AscendTensor& th) { return addOutput(th, th.getAclDataFormat()); }
    AclOpRunner& addOutputWithoutContiguous(diopiTensorHandle_t th) {
        AscendTensor at = AscendTensor(th);
        return addOutputWithoutContiguous(at);
    }

    /**
     * @brief: set acl op attribute.
     * @param[in] attrName attribute name.
     * @param[in] value attribute value.
     * @return A reference to the modified AclOpRunner.
     */
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
        ASCEND_CHECK_ABORT(false, "%s: no specialization for %s type.", dumpRunnerInfo().c_str(), typeid(T).name());
        return *this;
    }

    /**
     * @brief: set acl op attribute.
     * @param[in] attrName attribute name.
     * @param[in] value attribute value(vector).
     * @return A reference to the modified AclOpRunner.
     */
    template <typename T>
    AclOpRunner& setAttr(const std::string& attrName, const typename std::vector<T>& value) {
        std::vector<int64_t> vec(value.begin(), value.end());
        CALL_ACLRT(aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
        return *this;
    }

    /**
     * @brief: add output diopi tensor to AclOpRuner.
     * @param[out] th output tensor.
     * @param[in] format target tensor format for ascend
     * @param[in] dtype target tensor dtype
     * @return A reference to the modified AclOpRunner
     */
    AclOpRunner& addSyncOutput(AscendTensor& at, aclFormat format = ACL_FORMAT_UNDEFINED) {
        syncIdxs.push_back(outputIndex_);
        diopiTensorHandle_t th = static_cast<diopiTensorHandle_t>(at);
        syncTensors.push_back(&th);
        if (ACL_FORMAT_UNDEFINED == format) {
            addOutput(at, at.getAclDataFormat());
        } else {
            addOutput(at, format);
        }
        sync_ = true;
        return *this;
    }

    AclOpRunner& addSyncOutput(diopiTensorHandle_t* th, aclFormat format = ACL_FORMAT_UNDEFINED) {
        AscendTensor at = AscendTensor(*th);
        return addSyncOutput(at, format);
    }

    void preRun() { return; }

    void postRun(diopiStreamHandle_t stream) {
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
    }

    /**
     * @brief This function call the ascend executor. It is the core function in this class.
     */
    template <aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
    AclOpRunner& run() {
        diopiStreamHandle_t stream;
        diopiGetStream(context_, &stream);
        preRun();
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

        postRun(stream);
        return *this;
    }
};

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACLOPRUNNER_HPP_
