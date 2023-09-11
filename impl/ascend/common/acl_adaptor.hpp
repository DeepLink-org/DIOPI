#ifndef IMPL_ASCEND_COMMON_ACL_BRIDGE_HPP_
#define IMPL_ASCEND_COMMON_ACL_BRIDGE_HPP_

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "../ascend_tensor.hpp"
#include "../error.hpp"
#include "debug.hpp"
#include "promote_type.hpp"
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
 * AclAdaptor<2, 1>("xxx", ctx)
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
 * AclAdaptor<1, 1>("xxx", ctx)
 *     .addDynamicInput(vec)
 *     .addOutput(output)
 *     .run();
 * ```
 */
template <int InputSize, int OutputSize, aclDataType (*dtypeCastStrategy)(diopiDtype_t) = getAclDataType>
class AclAdaptor final {
private:
    std::string opname_;
    aclopAttr* attr_;
    std::vector<aclTensorDesc*> inputDescs_;
    std::vector<aclDataBuffer*> inputBuffers_;
    std::array<aclTensorDesc*, OutputSize> outputDescs_;
    std::array<aclDataBuffer*, OutputSize> outputBuffers_;
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
    explicit AclAdaptor(std::string opname, diopiContextHandle_t context) : context_(context), opname_(std::move(opname)), attr_(aclopCreateAttr()) {
        inputDescs_.resize(InputSize, nullptr);
        inputBuffers_.resize(InputSize, nullptr);
        outputDescs_.fill(nullptr);
        outputBuffers_.fill(nullptr);
    }

    ~AclAdaptor() {
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

    // TODO
    AclAdaptor& addConstInput() { return *this; }

    AclAdaptor& addInput(const AscendTensor& tensor, aclFormat format = ACL_FORMAT_UNDEFINED) {
        ASCEND_CHECK_ABORT(inputIndex_ >= 0 && inputIndex_ < inputSize(), "check 0<=inputIndex_<inputSize() failed");

        auto& desc = inputDescs_[inputIndex_];
        auto& buffer = inputBuffers_[inputIndex_];

        if (ACL_FORMAT_UNDEFINED == format) {
            format = tensor.getAclDataFormat();
        }
        desc = aclCreateTensorDesc(dtypeCastStrategy(tensor.dtype()), tensor.dim(), tensor.shape().data(), format);
        ASCEND_CHECK_ABORT(desc != nullptr, "aclTensorDesc should not be nullptr.");
        buffer = aclCreateDataBuffer(const_cast<void*>(tensor.data()), tensor.getBaseBufferSize());
        inputIndex_++;

        return *this;
    }

    /**
     * TODO: ~~可能要修改~~
     * @brief: add diopi tensor to AclOpRuner.
     * @param[in] ptr data pointer
     * @param[in] buffersize data length
     * @param[in] dims target tensor dims
     * @param[in] format target tensor format for ascend
     * @param[in] dtype target tensor dtype
     * @return A reference to the modified AclAdaptor
     */
    AclAdaptor& addInput(const void* ptr, int64_t buffersize, std::vector<int64_t>& dims, const aclFormat& format, diopiDtype_t dtype) {
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

    AclAdaptor& addInput(AscendTensor& tensor, const std::set<diopiDtype_t>& supportDtypes) {
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
        AscendTensor promoteTypeTensor = createAscendTensor(context_, &sizeTmp, nullptr, tensor.dtype(), tensor.device());

        return addInput(promoteTypeTensor);
    }

    /**
     * @brief: support input dynamic length tensors, need only one input(dynamic)
     * @param[in] tensors diopi const tensors pointer.
     * @return A reference to the modified AclAdaptor
     */
    template <typename T>
    AclAdaptor& addDynamicInput(const std::vector<T>& tensors) {
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

    void preRun() { return; }

    void postRun(diopiStreamHandle_t stream) {
        // for (auto pair : nonContiguousOutputPairs) {
        //     auto th = pair.first;
        //     auto thCopy = pair.second;
        //     diopiCopyInp(context_, thCopy, th);
        // }
        // for (int64_t i = 0; i < syncIdxs.size(); i++) {
        //     auto syncIdx = syncIdxs[i];
        //     auto syncTensorPtr = syncTensors[i];
        //     int descNumDims = aclGetTensorDescNumDims(outputDescs_[syncIdx]);
        //     std::vector<int64_t> realShape;
        //     int64_t dimSize = 0;
        //     for (int64_t j = 0; j < descNumDims; j++) {
        //         CALL_ACLRT(aclGetTensorDescDimV2(outputDescs_[syncIdx], j, &dimSize));
        //         realShape.push_back(dimSize);
        //     }
        //     diopiTensorHandle_t syncTensorReal;
        //     diopiSize_t syncTensorRealSize = vectorToDiopiSize(realShape);
        //     diopiDtype_t dtype;
        //     diopiGetTensorDtype(*syncTensorPtr, &dtype);
        //     diopiRequireTensor(context_, &syncTensorReal, &syncTensorRealSize, nullptr, dtype, diopi_device);
        //     int64_t elemsize, numel, buffersize;
        //     diopiGetTensorElemSize(syncTensorReal, &elemsize);
        //     diopiGetTensorNumel(syncTensorReal, &numel);
        //     void *dst, *src;
        //     diopiGetTensorData(*syncTensorPtr, &src);
        //     diopiGetTensorData(syncTensorReal, &dst);
        //     buffersize = numel * elemsize;
        //     if (buffersize > 0 && src != nullptr && dst != nullptr) {
        //         CALL_ACLRT(aclrtMemcpyAsync(dst, buffersize, src, buffersize, ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        //     }
        //     *syncTensorPtr = syncTensorReal;
        // }
        // CALL_ACLRT(aclrtSynchronizeStream(stream));
        // // Get environment variables once when run is called for the first time
        // static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
        // if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
        //     info(dumpRunnerInfo().c_str());
        // }
    }

    /**
     * @brief This function call the ascend executor. It is the core function in this class.
     */
    template <aclEngineType EngineType = ACL_ENGINE_SYS, aclCompileType CompileType = ACL_COMPILE_SYS>
    AclAdaptor& run() {
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

#endif  //  IMPL_ASCEND_COMMON_ACL_BRIDGE_HPP_
