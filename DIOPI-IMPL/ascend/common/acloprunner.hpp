#ifndef ACLOPRUNNER_HPP_
#define ACLOPRUNNER_HPP_

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <utility>
#include <deque>
#include <iostream>
#include <string>
#include <vector>
#include <initializer_list>
#include <algorithm>

#include <diopi/diopirt.h>

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
        case diopi_dtype_float16: return ACL_FLOAT16;
        case diopi_dtype_float32: return ACL_FLOAT;
        case diopi_dtype_float64: return ACL_DOUBLE;
        case diopi_dtype_int8: return ACL_INT8;
        case diopi_dtype_uint8: return ACL_UINT8;
        case diopi_dtype_int16: return ACL_INT16;
        case diopi_dtype_uint16: return ACL_UINT16;
        case diopi_dtype_int32: return ACL_INT32;
        case diopi_dtype_uint32: return ACL_UINT32;
        case diopi_dtype_int64: return ACL_INT64;
        case diopi_dtype_uint64: return ACL_UINT64;
        case diopi_dtype_bool: return ACL_BOOL;
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
        /*
        TODO：support different memformat
        if (spec.probableMemoryFormat() == MemoryFormat::ChannelsLast) {
            return ACL_FORMAT_NHWC;
        } else {
            return ACL_FORMAT_NCHW;
        }
        */
       return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}


class AclTensorDesc final{
    aclTensorDesc* desc_ = nullptr;

public:
    explicit AclTensorDesc(const diopiTensorHandle_t& th, const aclFormat& format) {
        diopiSize_t shape;
        diopiSize_t stride;
        diopiGetTensorShape(th, &shape);
        diopiGetTensorStride(th, &stride);
        int64_t numel = 0;
        diopiGetTensorNumel(th, &numel);

        std::vector<int64_t> dims(shape.len);
        for (size_t i = 0; i < dims.size(); ++i) {
            dims[i] = shape.data[i];
        }
        if (dims.size() == 0 && numel == 1) {
            dims.push_back(1);
        }

        desc_ = aclCreateTensorDesc(getAclDataType(th),
            dims.size(), dims.data(), format);
        check_args(desc_ != nullptr, "aclTensorDesc should not be nullptr.");
    }

    AclTensorDesc(AclTensorDesc &&other) = delete;

    AclTensorDesc(const AclTensorDesc& other) = delete;

    AclTensorDesc& operator=(const AclTensorDesc&) = delete;

    AclTensorDesc& operator=(AclTensorDesc&& other) = delete;

    ~AclTensorDesc() {
        if (desc_) {
            aclDestroyTensorDesc(desc_);
        }
        desc_ = nullptr;
    }

    aclTensorDesc* get() { return desc_; }
};

//template<const char* OpName, int InputSize, int OutputSize>
class AclOpRunner final{
private:
    std::string opname_;
    std::deque<AclTensorDesc> inputDesc_;
    std::deque<AclTensorDesc> outputDesc_;
    std::vector<aclDataBuffer*> inputBufferPtrs_;
    std::vector<aclDataBuffer*> outputBufferPtrs_;
    aclopAttr* attr_;

    std::string dumpRunnerInfo();

public:
    explicit AclOpRunner(std::string opname)
            : opname_(std::move(opname)), attr_(aclopCreateAttr()) {
        check_args(attr_ != nullptr, "aclopAttr* attr_ shoule not be nullptr.");
    }

    AclOpRunner(const AclOpRunner& other) = delete;

    AclOpRunner(AclOpRunner&& other) = delete;

    AclOpRunner& operator=(const AclOpRunner&) = delete;

    AclOpRunner& operator=(AclOpRunner&& other) = delete;

    ~AclOpRunner() {
        aclopDestroyAttr(attr_);
        std::for_each(inputBufferPtrs_.begin(), inputBufferPtrs_.end(),
            [](aclDataBuffer *buff) { CALL_ACLRT(aclDestroyDataBuffer(buff));});

        std::for_each(outputBufferPtrs_.begin(), outputBufferPtrs_.end(),
            [](aclDataBuffer* buff) { CALL_ACLRT(aclDestroyDataBuffer(buff));});
        inputDesc_.clear();
        outputDesc_.clear();
        inputBufferPtrs_.clear();
        outputBufferPtrs_.clear();
    }

    AclOpRunner& addInput(const diopiTensorHandle_t& th, const aclFormat& format) {
        inputDesc_.emplace_back(th, format);
        const void* ptr = nullptr;
        diopiGetTensorDataConst(th, &ptr);

        int64_t numel = 0;
        int64_t itemsize = 0;
        diopiGetTensorNumel(th, &numel);
        diopiGetTensorElemSize(th, &itemsize);

        inputBufferPtrs_.emplace_back(aclCreateDataBuffer(
            const_cast<void*>(ptr), numel * itemsize));
        diopiDevice_t device;
        diopiGetTensorDevice(th, &device);
        if (device == diopi_host) {
            CALL_ACLRT(aclSetTensorPlaceMent(inputDesc_.back().get(), ACL_MEMTYPE_HOST));
        }
        return *this;
    }

    AclOpRunner& addInput(const diopiTensorHandle_t& th) {
        return addInput(th, getAclDataFormat(th));
    }

    /*
    template<typename T>
    AclOpRunner& addInput(const std::vector<T>& in) {
        inputDesc_.emplace_back(DArraySpec::array(PrimMap<T>::get(), in.size()), ACL_FORMAT_ND);
        CALL_ACLRT(aclSetTensorPlaceMent(inputDesc_.back().get(), ACL_MEMTYPE_HOST));

        inputBufferPtrs_.emplace_back(aclCreateDataBuffer(
            const_cast<T*>(in.data()), in.size() * sizeof(T)));
        return *this;
    }
    */

    template <typename T, typename... Ins>
    AclOpRunner& addInput(const T& in, const Ins&... ins) {
        return addInput(in).addInput(ins...);
    }

    AclOpRunner& addOutput(diopiTensorHandle_t& th) {
        void* ptr = nullptr;
        diopiGetTensorData(th, &ptr);

        int64_t numel = 0;
        int64_t itemsize = 0;
        diopiGetTensorNumel(th, &numel);
        diopiGetTensorElemSize(th, &itemsize);
        outputDesc_.emplace_back(th, getAclDataFormat(th));
        outputBufferPtrs_.emplace_back(aclCreateDataBuffer(
            ptr, numel * itemsize));
        return *this;
    }

    template <typename... Outs>
    AclOpRunner& addOutput(diopiTensorHandle_t& out, Outs&... outs) {
        return addOutput(out).addOutput(outs...);
    }

    template<typename T>
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

        check_args(false,"no specialization for this type.");
    }

    AclOpRunner& run(diopiContextHandle_t& ctx);
};

template<>
AclOpRunner& AclOpRunner::setAttr<typename std::vector<size_t>>(
    const std::string& attrName, const std::vector<size_t>& value);

template<>
AclOpRunner& AclOpRunner::setAttr<typename std::vector<int64_t>>(
    const std::string& attrName, const std::vector<int64_t>& value);

template<>
AclOpRunner& AclOpRunner::setAttr<typename std::vector<int32_t>>(
    const std::string& attrName, const std::vector<int32_t>& value);

template<>
AclOpRunner& AclOpRunner::setAttr<std::string>(
    const std::string& attrName, const std::string& value);

}  // namespace ascend
}  // namespace impl


#endif  //  ACLOPRUNNER_HPP_