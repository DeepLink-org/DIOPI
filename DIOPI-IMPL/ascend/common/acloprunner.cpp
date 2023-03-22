// Copyright (c) 2022, SenseTime.
#include "acloprunner.hpp"
#include <acl/acl_op_compiler.h>
#include <vector>
#include <sstream>

namespace impl {
namespace ascend {

/*
template<>
AclOpRunner& AclOpRunner::setAttr(const std::string& attrName, const DArrayShape& value) {
    std::vector<int64_t> vec(value.ndims());
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = value.dim(i);
    }
    CALL_ACLRT(
        aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
    return *this;
}
*/

template<>
AclOpRunner& AclOpRunner::setAttr<typename std::vector<size_t>>(const std::string& attrName,
        const std::vector<size_t>& value) {
    std::vector<int64_t> vec(value.begin(), value.end());
    CALL_ACLRT(
        aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
    return *this;
}


template<>
AclOpRunner& AclOpRunner::setAttr<typename std::vector<int32_t>>(
        const std::string& attrName, const std::vector<int32_t>& value) {
    std::vector<int64_t> vec(value.begin(), value.end());
    CALL_ACLRT(
        aclopSetAttrListInt(attr_, attrName.data(), vec.size(), vec.data()));
    return *this;
}


template<>
AclOpRunner& AclOpRunner::setAttr(const std::string& attrName, const std::vector<int64_t>& value) {
    CALL_ACLRT(aclopSetAttrListInt(
        attr_, attrName.data(), value.size(), value.data()));
    return *this;
}

template<>
AclOpRunner& AclOpRunner::setAttr(const std::string& attrName, const std::string& value) {
    CALL_ACLRT(aclopSetAttrString(attr_, attrName.data(), value.data()));
    return *this;
}


std::string AclOpRunner::dumpRunnerInfo() {
    std::stringstream sstream;
    sstream << "opname:" << opname_ << ",ins.size:" << inputBufferPtrs_.size()
            << ",outs.size:" << outputBufferPtrs_.size() << std::endl;
    return sstream.str();
}


AclOpRunner& AclOpRunner::run(diopiContextHandle_t& ctx) {
    diopiStreamHandle_t stream;
    diopiGetStream(ctx, &stream);
    std::vector<aclTensorDesc*> inDescs;
    inDescs.reserve(inputDesc_.size());
    for (auto& desc : inputDesc_) {
        inDescs.push_back(desc.get());
    }

    std::vector<aclTensorDesc*> outDescs;
    outDescs.reserve(outputDesc_.size());
    for (auto& desc : outputDesc_) {
        outDescs.push_back(desc.get());
    }

    auto errorcode = aclopCompileAndExecute(opname_.data(), inputBufferPtrs_.size(),
            inDescs.data(), inputBufferPtrs_.data(), outputBufferPtrs_.size(), outDescs.data(),
            outputBufferPtrs_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS,
            nullptr, stream);

    check_args(errorcode == ACL_SUCCESS, dumpRunnerInfo().c_str());

    //  Get environment variables once when run is called for the first time
    static int PARROTS_DEBUG_ACLOPRUNNER = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
    if (PARROTS_DEBUG_ACLOPRUNNER > 0) {
        info(dumpRunnerInfo().c_str());
    }

    return *this;
}

}  // namespace ascend
}  // namespace impl
