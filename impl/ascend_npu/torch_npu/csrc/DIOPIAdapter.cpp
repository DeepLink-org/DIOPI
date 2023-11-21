#include "torch_npu/csrc/framework/DIOPIAdapter.h"

namespace at_npu {
namespace native {

UnifiedResult OpPreparation::binary_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, bool check_mem_overlap) {
    UnifiedResult unified_result;
    TORCH_CHECK(a.dtype() == b.dtype());
    return unified_result;
}

UnifiedResult OpPreparation::binary_op_check(at::Tensor &out, const at::Tensor &a, const c10::Scalar b, bool check_mem_overlap) {
    UnifiedResult unified_result;
    TORCH_CHECK(a.dtype() == b.type());
    return unified_result;
}

// NOTE [Check Match for Npu at::Tensor]
// check_match is used to ensure that npu tensor satisfies the
// calculation requirements of npu operators.
// The rules are as follows,
// 1、tensor should be contiguous
// Not contiguous means the operator needs to read and write memory
// at intervals according to strides and sizes. Npu operators has
// no such ability for the time being
// 2、metadata should be match
// Resize_ a contiguous cpu tensor from [1,2,3,4] to [4,3,2,1] no
// need to change the physical memory. However, for a contiguous npu
// tensor whose npu_format_ is 5HD, storage shape should be change
// from [1,1,3,4,16] to [4,1,2,1,16]. So metadata not match often
// results in unexpected physical memory. format_contiguous will be
// called preparing correct memory of operand in these case.
bool NpuUtils::check_match(const at::Tensor *tensor) {
    // case1:uncontiguous tensor
    if (!tensor->is_contiguous()) {
        return false;
    }

#if 0
  // case2:meta data not match, sizes or strides of presentation
  // layer is different from that of storage layer
  if (!StorageDescHelper::MetaDataAreMatch(tensor)) {
    return false;
  }
  // case3:meta data not match, storage_offset of presentation layer
  // is different from that of storage layer
  bool isPadding = FormatHelper::IsPadded(tensor);
  if (isPadding && (!StorageDescHelper::OffsetAreMatch(tensor))) {
    return false;
  }
#endif
    return true;
}

at::Tensor NpuUtils::format_contiguous_add_copy_optimize(const at::Tensor &src) {
    // case1:tensor src is not contiguous
    if (!src.is_contiguous()) {
        return src.contiguous();
    }
#if 0
  // case2:meta data not match, sizes or strides of presentation
  // layer is different from that of storage layer
  if (!StorageDescHelper::MetaDataAreMatch(&src)) {
    // Fix not match case2, tensor should have matched metadata and
    // NPUStorageDesc.
    RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
    return metadata_convert_match_with_copy_optimize(src);
  }

  // case3:meta data not match, storage_offset of presentation layer
  // is different from that of storage layer
  if (FormatHelper::IsPadded(&src) &&
      (!StorageDescHelper::OffsetAreMatch(&src))) {
    // Fix not match case3, tensor with padding should not have storage-offset.
    RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
    return metadata_with_offset_padding_convert_match(src);
  }
#endif
    return src;
}

float CalcuOpUtil::GetScalarFloatValue(const c10::Scalar &scalar) {
    float value;
    if (scalar.isFloatingPoint()) {
        value = scalar.toFloat();
    } else {
        value = (float)scalar.toInt();
    }

    return value;
}

int64_t CalcuOpUtil::GetTensorNpuFormat(const at::Tensor &tensor) {
    int64_t ndim = tensor.sizes().size();
    if (ndim == 5) {
        return ACL_FORMAT_NCDHW;
    } else if (ndim == 4) {
        return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}

// OpPreparation part
void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst) {
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), dst.sizes());
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst, c10::IntArrayRef shape) {
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), shape);
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor> &input, at::Tensor &output, int64_t format, at::ScalarType dtype, c10::IntArrayRef shape) {
    // Check that the outputs have no internal overlap
    // and do not share memory with inputs.
    c10::SmallVector<at::Tensor, N> inputs{input};
    c10::SmallVector<at::Tensor, N> outputs = {output};
    CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
    TORCH_CHECK(at_npu::key::isDeviceTensor(output), "output with device ", output.device(), " doesn't match the desired device NPU");
    TORCH_CHECK(output.scalar_type() == dtype, "expected dtype ", dtype, " but got dtype ", output.scalar_type());

    bool is_read_write = false;
    // check if output is also an input
    for (const auto &input : inputs) {
        if (output.is_same(input)) {
            is_read_write = true;
            break;
        }
    }

    // Preserve legacy resizing behavior of out=... arguments
    if (!output.sizes().equals(shape)) {
        TORCH_CHECK(!is_read_write, "output with shape ", output.sizes(), " doesn't match the broadcast shape ", shape);
        output.resize_(shape);
    }

    if (CalcuOpUtil::GetTensorNpuFormat(output) != format) {
        TORCH_CHECK(!is_read_write, "can not cast format when output is input");
        NPUNativeFunctions::npu_format_cast_(output, format);
    }
}

class OpAttrMaker {
public:
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, bool value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, int64_t value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, float value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, string value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, c10::IntArrayRef value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, at::ArrayRef<uint8_t> value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, c10::Scalar value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, at::ScalarType value);
    TORCH_NPU_API static void Set(aclopAttr *attr, const string &name, at::ArrayRef<c10::IntArrayRef> value);
};  // class OpAttrMaker

void OpAttrMaker::Set(aclopAttr *attr, const string &name, bool value) { aclopSetAttrBool(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, int64_t value) { aclopSetAttrInt(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, float value) { aclopSetAttrFloat(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, string value) { aclopSetAttrString(attr, name.c_str(), value.c_str()); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::IntArrayRef value) { aclopSetAttrListInt(attr, name.c_str(), value.size(), value.data()); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<float> value) { aclopSetAttrListFloat(attr, name.c_str(), value.size(), value.data()); }

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<uint8_t> value) {
    aclopSetAttrListBool(attr, name.c_str(), value.size(), value.data());
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, c10::Scalar value) {
    float val = CalcuOpUtil::GetScalarFloatValue(value);
    aclopSetAttrFloat(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ScalarType value) {
    aclDataType val = CalcuOpUtil::ConvertToAclDataType(value);
    aclopSetAttrDataType(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr *attr, const string &name, at::ArrayRef<c10::IntArrayRef> value) {
    // Pointer to values of each listInt.
    c10::SmallVector<int64_t *, N> attrValue;
    // Pointer to number of each listInt.
    c10::SmallVector<int, N> eachListIntNum;
    // Value of each listInt.
    c10::SmallVector<c10::SmallVector<int64_t, N>, N> eachListIntVal;
    for (const auto i : c10::irange(value.size())) {
        c10::SmallVector<int64_t, N> listInt;
        int64_t valueSize = static_cast<int64_t>(value[i].size());
        listInt.resize(valueSize);
        std::copy(value[i].begin(), value[i].end(), listInt.begin());
        eachListIntVal.emplace_back(listInt);
        attrValue.emplace_back(eachListIntVal.back().data());
        eachListIntNum.emplace_back(valueSize);
    }

    aclopSetAttrListListInt(attr, name.c_str(), attrValue.size(), eachListIntNum.data(), attrValue.data());
}

class AclTensorDescMaker {
public:
    AclTensorDescMaker() {}
    ~AclTensorDescMaker() = default;

    AclTensorDescMaker &Create(aclDataType dataType, torch_npu::NPUStorageDesc storageDesc) {
        c10::SmallVector<int64_t, 5> dims;
        // if aclDataType is ACL_STRING, storageDims is empty.
        if (dataType != ACL_STRING) {
            dims = storageDesc.base_sizes_;
        }
        auto format = storageDesc.origin_format_;
        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker &Create(aclDataType dataType, c10::IntArrayRef dims, aclFormat format) {
        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker &Create(aclDataType dataType, aclFormat format) {
        desc = aclCreateTensorDesc(dataType, 0, nullptr, format);
        return *this;
    }

    inline AclTensorDescMaker &SetFormat(aclFormat format) {
        aclSetTensorFormat(desc, format);
        return *this;
    }

    inline AclTensorDescMaker &SetPlacement(aclMemType memType) {
        aclSetTensorPlaceMent(desc, memType);
        return *this;
    }

    template <unsigned int N>
    inline AclTensorDescMaker &SetShape(const c10::SmallVector<int64_t, N> &dims) {
        aclSetTensorShape(desc, dims.size(), dims.data());
        return *this;
    }

    template <unsigned int N>
    AclTensorDescMaker &SetRange(const c10::SmallVector<int64_t, N> &rangs) {
        int arryDim = rangs.size() == 0 ? 0 : rangs.size() / 2;

        int64_t range[arryDim][2];
        for (int i = 0, j = 0; i < arryDim; i++, j += 2) {
            range[i][0] = rangs[j];
            range[i][1] = rangs[j + 1];
        }

        aclSetTensorShapeRange(desc, arryDim, range);
        return *this;
    }

    inline AclTensorDescMaker &SetName(const std::string &name) {
        if (!name.empty()) {
            aclSetTensorDescName(desc, name.c_str());
        }
        return *this;
    }

    inline AclTensorDescMaker &SetConstAttr(c10::optional<at::Tensor> cpu_tensor) {
        if (cpu_tensor.has_value() && cpu_tensor.value().defined()) {
            aclSetTensorConst(desc, cpu_tensor.value().data_ptr(), cpu_tensor.value().itemsize() * cpu_tensor.value().numel());
        }

        return *this;
    }

    inline aclTensorDesc *Get() const { return desc; }

private:
    aclTensorDesc *desc = nullptr;
};  // class AclTensorDescMaker

//
class AclTensorBufferMaker {
public:
    // base of Ctr
    // params: tensor, offset, remained size
    AclTensorBufferMaker(const at::Tensor *tensor, int64_t offset, int64_t n) {
        uint8_t *header = reinterpret_cast<uint8_t *>(tensor->data_ptr()) - tensor->itemsize() * static_cast<uint8_t>(offset);
        size_t bufferSize = tensor->itemsize() * static_cast<size_t>(n);
        ptr = aclCreateDataBuffer(header, bufferSize);
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor *tensor, int64_t n = 1) {
        if (tensor == nullptr || n == 0) {
            ptr = aclCreateDataBuffer(nullptr, 0);
        } else {
            ptr = aclCreateDataBuffer((void *)(tensor->data_ptr()), tensor->itemsize() * n);
        }
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor &tensor, int64_t n = 1) { ptr = aclCreateDataBuffer((void *)(tensor.data_ptr()), tensor.itemsize() * n); }

    ~AclTensorBufferMaker() = default;

    inline aclDataBuffer *Get() const { return ptr; }

private:
    aclDataBuffer *ptr = nullptr;
};  // class AclTensorBufferMaker

struct ACL_PARAMS {
    ACL_PARAMS() {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
    }

    int input_num{0};
    const aclTensorDesc **input_desc;
    const aclDataBuffer **input_data_buf;
    int output_num{0};
    const aclTensorDesc **output_desc;
    aclDataBuffer **output_data_buf;
};

struct ACL_DYNAMIC_PARAMS {
    ACL_DYNAMIC_PARAMS() {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
        inputDims = nullptr;
        outputDims = nullptr;
        inputFormats = nullptr;
        outputFormats = nullptr;
        compile_input_desc = nullptr;
        compile_output_desc = nullptr;

        hasAttr = false;
    }

    int input_num = 0;
    const aclTensorDesc **input_desc;
    const aclDataBuffer **input_data_buf;
    int output_num = 0;
    const aclTensorDesc **output_desc;
    aclDataBuffer **output_data_buf;
    int64_t *inputDims;
    int64_t *outputDims;
    aclFormat *inputFormats;
    aclFormat *outputFormats;
    const aclTensorDesc **compile_input_desc;
    const aclTensorDesc **compile_output_desc;
    bool hasAttr;
    std::string dynamicKey;
};

struct CONST_PARAMS {
    int constNum = 0;
    const int64_t **constList = nullptr;
    const int64_t *constIdx = nullptr;
    CONST_PARAMS() = default;
};

struct ExecuteParas {
    using PROCESS_FUNC = std::function<int()>;
    char opType[50]{};
    bool isJitDisable = false;
    ACL_PARAMS paras;
    CONST_PARAMS constParams;
    const aclopAttr *attr;
    int64_t constIdx = -1;
    static std::atomic<uint64_t> g_pta_correlation_id;
    uint64_t pta_correlation_id = 0;
    c10::SmallVector<at::Tensor, N> hostMemory;
    ExecuteParas() = default;
    void Release();
    void Copy(ExecuteParas &other);
    void CopyEx(ExecuteParas &other);
    PROCESS_FUNC customHandler;
};

NPUStatus DestroyAclParams(ACL_PARAMS &params);
void DestroyConstParams(CONST_PARAMS &params);

std::atomic<uint64_t> ExecuteParas::g_pta_correlation_id{0};
void ExecuteParas::Release() {
    // if useDynamicCompile, this attr will be freed in dynamic compile.
    if (attr != nullptr) {
        aclopDestroyAttr(attr);
    }
    DestroyConstParams(constParams);
    NPUStatus ret = DestroyAclParams(paras);
    if (ret != SUCCESS) {
        NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
    }
    hostMemory.clear();
    customHandler = nullptr;
    return;
}

void ExecuteParas::Copy(ExecuteParas &other) {
    strncpy(this->opType, other.opType, sizeof(ExecuteParas::opType) - 1);
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
    this->hostMemory = other.hostMemory;
    this->isJitDisable = other.isJitDisable;
    this->customHandler = other.customHandler;
    this->pta_correlation_id = other.pta_correlation_id;
}

void ExecuteParas::CopyEx(ExecuteParas &other) {
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
}

NPUStatus DestroyAclParams(ACL_PARAMS &params) {
    if (params.input_num != 0) {
        if (params.input_desc != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                aclDestroyTensorDesc(params.input_desc[i]);
            }
        }
        if (params.input_data_buf != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                NPU_CHECK_ERROR(aclDestroyDataBuffer(params.input_data_buf[i]));
            }
        }
        params.input_num = 0;
    }
    if (params.output_num != 0) {
        if (params.output_desc != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                aclDestroyTensorDesc(params.output_desc[i]);
            }
        }
        if (params.output_data_buf != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                NPU_CHECK_ERROR(aclDestroyDataBuffer(params.output_data_buf[i]));
            }
        }
        params.output_num = 0;
    }
    free(params.input_desc);
    params.input_desc = nullptr;
    params.input_data_buf = nullptr;
    params.output_desc = nullptr;
    params.output_data_buf = nullptr;
    return SUCCESS;
}

void DestroyConstParams(CONST_PARAMS &params) {
    if (params.constList != nullptr) {
        for (int i = 0; i < params.constNum; ++i) {
            if (params.constList[i] != nullptr) {
                delete[] params.constList[i];
            }
        }
    }
    params.constList = nullptr;
    params.constIdx = nullptr;
}

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr
class OpCommandImpl {
public:
    OpCommandImpl() {}
    ~OpCommandImpl() {
        // do nothing, can not release resource, because of multi-thread or
        // queue-enable
    }

    void SetName(const string &name) { opName = name; }

    void SetCustomHandler(PROC_FUNC func) { execParam.customHandler = func; }

    const string &GetName() const { return opName; }

    void AddInput(const aclTensorDesc *desc, const aclDataBuffer *buffer) {
        execParam.inDesc.emplace_back(std::move(desc));
        execParam.inBuffer.emplace_back(std::move(buffer));
    }

    void AddInput(const aclTensorDesc *desc, const aclDataBuffer *buffer, const at::Tensor &hostTensor) {
        AddInput(desc, buffer);
        execParam.hostMem.emplace_back(hostTensor);
    }

    void AddInput(const string &str);

    void AddOutput(const aclTensorDesc *desc, aclDataBuffer *buffer) {
        execParam.outDesc.emplace_back(std::move(desc));
        execParam.outBuffer.emplace_back(std::move(buffer));
    }

    template <typename dataType>
    void AddAttr(const string &attrName, dataType value) {
        InitAttr();
        OpAttrMaker::Set(execParam.attr, attrName, value);
    }

    // export op execute params
    void ExportParams(ExecuteParas &params) {
        TORCH_CHECK(sizeof(ExecuteParas::opType) >= opName.length() + 1, "Too long Ascend IR Name: ", opName);
        memset(params.opType, '\0', sizeof(params.opType));
        opName.copy(params.opType, opName.length() + 1);
        params.attr = execParam.attr;
        // make params
        int inputNum = static_cast<int>(execParam.inDesc.size());
        int outputNum = static_cast<int>(execParam.outDesc.size());

        size_t inputTensorDescArrLen = inputNum * sizeof(uintptr_t);
        size_t inputDataBuffArrLen = inputNum * sizeof(uintptr_t);

        size_t outputTensorDescArrLen = outputNum * sizeof(uintptr_t);
        size_t outputDataBuffArrLen = outputNum * sizeof(uintptr_t);

        size_t totalMemLen = inputTensorDescArrLen + inputDataBuffArrLen + outputTensorDescArrLen + outputDataBuffArrLen;

        char *basePtr = static_cast<char *>(malloc(totalMemLen));
        AT_ASSERT(basePtr != nullptr);
        const aclTensorDesc **aclTensorInputDescArr = reinterpret_cast<const aclTensorDesc **>(basePtr);
        basePtr += inputTensorDescArrLen;
        const aclDataBuffer **aclDataInputBuffArr = reinterpret_cast<const aclDataBuffer **>(basePtr);
        basePtr += inputDataBuffArrLen;

        const aclTensorDesc **aclTensorOutputDescArr = reinterpret_cast<const aclTensorDesc **>(basePtr);
        basePtr += outputTensorDescArrLen;
        aclDataBuffer **aclDataOutputBuffArr = reinterpret_cast<aclDataBuffer **>(basePtr);

        std::copy(execParam.inDesc.begin(), execParam.inDesc.end(), aclTensorInputDescArr);
        std::copy(execParam.inBuffer.begin(), execParam.inBuffer.end(), aclDataInputBuffArr);
        std::copy(execParam.outDesc.begin(), execParam.outDesc.end(), aclTensorOutputDescArr);
        std::copy(execParam.outBuffer.begin(), execParam.outBuffer.end(), aclDataOutputBuffArr);

        params.paras.input_num = inputNum;
        params.paras.output_num = outputNum;
        params.paras.input_desc = aclTensorInputDescArr;
        params.paras.input_data_buf = aclDataInputBuffArr;
        params.paras.output_desc = aclTensorOutputDescArr;
        params.paras.output_data_buf = aclDataOutputBuffArr;
        params.hostMemory = execParam.hostMem;
        params.customHandler = execParam.customHandler;
        params.pta_correlation_id = ExecuteParas::g_pta_correlation_id++;
#if 0
        if (!ForceJitCompileList::GetInstance().Inlist(opName) && env::CheckJitDisable()) {
            params.isJitDisable = true;
        }
#endif
    }

    // Set engine priority for op on data preprocessing stream
    void SetEnginePriority();

    void Run(bool sync, c10::SmallVector<int64_t, N> &sync_index, c10::SmallVector<at::Tensor, N> &outputTensor);

    void releaseSource(bool no_blocking = true) {
        if (no_blocking) {
            std::for_each(execParam.inDesc.begin(), execParam.inDesc.end(), aclDestroyTensorDesc);
            std::for_each(execParam.outDesc.begin(), execParam.outDesc.end(), aclDestroyTensorDesc);
            std::for_each(execParam.inBuffer.begin(), execParam.inBuffer.end(), aclDestroyDataBuffer);
            std::for_each(execParam.outBuffer.begin(), execParam.outBuffer.end(), aclDestroyDataBuffer);
            if (execParam.attr != nullptr) {
                aclopDestroyAttr(execParam.attr);
                execParam.attr = nullptr;
            }
        }

        execParam.inDesc.clear();
        execParam.inBuffer.clear();

        execParam.outDesc.clear();
        execParam.outBuffer.clear();

        execParam.hostMem.clear();

        // recover
        execParam.attr = nullptr;
        execParam.customHandler = nullptr;
        opName = "";
    }

private:
    struct AclExecParam {
        c10::SmallVector<const aclTensorDesc *, N> inDesc;    // owned
        c10::SmallVector<const aclDataBuffer *, N> inBuffer;  // owned
        c10::SmallVector<const aclTensorDesc *, N> outDesc;   // owned
        c10::SmallVector<aclDataBuffer *, N> outBuffer;       // owned
        c10::SmallVector<at::Tensor, N> hostMem;
        aclopAttr *attr = nullptr;
        PROC_FUNC customHandler = nullptr;
    };

    void InitAttr() {
        if (execParam.attr == nullptr) {
            execParam.attr = aclopCreateAttr();
        }
    }

    aclError InnerRun(const string &name, AclExecParam &params, bool sync, c10::SmallVector<int64_t, N> &sync_index,
                      c10::SmallVector<at::Tensor, N> &outputTensor);

private:
    string opName;
    AclExecParam execParam;
};  // class OpCommandImpl

OpCommand::OpCommand(){INTERFACE_NOT_IMPL}

OpCommand::~OpCommand(){INTERFACE_NOT_IMPL}

OpCommand &OpCommand::Name(const string &name) {
    INTERFACE_NOT_IMPL
    return *this;
}

void OpCommand::SetCustomHandler(PROC_FUNC func){INTERFACE_NOT_IMPL}

OpCommand &OpCommand::Expect(UnifiedResult unified_result) {
    INTERFACE_NOT_IMPL
    return *this;
}

// None Input
OpCommand &OpCommand::Input() {
    INTERFACE_NOT_IMPL
    return *this;
}

// Tensor Input which need contiguous
OpCommand &OpCommand::Input(const at::Tensor &input, const string &descName, const c10::optional<aclFormat> &sensitive_format, const string &realData) {
    INTERFACE_NOT_IMPL
    return *this;
}

// IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
OpCommand &OpCommand::Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType, CompileType compileType, const string &realDtype,
                            const string &descName) {
    INTERFACE_NOT_IMPL
    return *this;
}

// Scalar Input, we will do h2d in launch kernel
OpCommand &OpCommand::Input(const c10::Scalar &input, const at::ScalarType type, CompileType compileType) {
    INTERFACE_NOT_IMPL
    return *this;
}

// Tensor Input which no need contiguous
OpCommand &OpCommand::InputWithoutContiguous(const at::Tensor &input, const string &descName, const string &realData) {
    INTERFACE_NOT_IMPL
    return *this;
}

// Output Tensor
OpCommand &OpCommand::Output(at::Tensor &output, const string &descName, const c10::optional<aclFormat> &sensitive_format, const string &realType) {
    INTERFACE_NOT_IMPL
    return *this;
}

// Run a single op
void OpCommand::Run(){INTERFACE_NOT_IMPL}

OpCommand &OpCommand::Sync(c10::SmallVector<int64_t, N> &index) {
    INTERFACE_NOT_IMPL
    return *this;
}

OpCommand &OpCommand::Sync() {
    INTERFACE_NOT_IMPL
    // c10_npu::NPUStream stream();
    // NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    return *this;
}

}  // namespace native
}  // namespace at_npu

namespace c10_npu {
namespace acl {

const char *AclGetErrMsg() {
    typedef const char *(*aclGetErrMsg)();
    static aclGetErrMsg func = aclGetRecentErrMsg;
    if (func != nullptr) {
        auto res = func();
        return res != nullptr ? res : "";
    }
    return "";
}
}  // namespace acl
}  // namespace c10_npu