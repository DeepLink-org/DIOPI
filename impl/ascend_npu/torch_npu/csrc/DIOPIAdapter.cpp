#include "torch_npu/csrc/framework/DIOPIAdapter.h"

#include <diopi/diopirt.h>


namespace {
constexpr float EPSILON = 1e-6;

// check all at::ScalarType is not negative
#define ENUM_PAIR_FUNC(_1, n) static_assert(static_cast<int64_t>(at::ScalarType::n) >= 0, #n " is negative");
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUM_PAIR_FUNC)
#undef ENUM_PAIR_FUNC

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
    _(at::ScalarType::Byte, ACL_UINT8)               \
    _(at::ScalarType::Char, ACL_INT8)                \
    _(at::ScalarType::Short, ACL_INT16)              \
    _(at::ScalarType::Int, ACL_INT32)                \
    _(at::ScalarType::Long, ACL_INT64)               \
    _(at::ScalarType::Half, ACL_FLOAT16)             \
    _(at::ScalarType::Float, ACL_FLOAT)              \
    _(at::ScalarType::Double, ACL_DOUBLE)            \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)    \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)   \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL)                \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::BFloat16, ACL_BF16)            \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// check at::ScalarType has been changed or not
#define ENUM_PAIR_FUNC(at_dtype, acl_dtype)                                                         \
    static_assert(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_dtype)] == (acl_dtype), \
                  #at_dtype " and " #acl_dtype                                                      \
                            " is not match any more, please check "                                 \
                            "AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR and modify it");
AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(ENUM_PAIR_FUNC)
#undef DEFINE_ENUM

static std::map<const string, const aclDataType> STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
    {"uint16", ACL_UINT16}, {"uint8", ACL_UINT8}, {"uint64", ACL_UINT64}, {"string", ACL_STRING}};

aclError AclrtMemcpyAsyncParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind, aclrtStream stream) {
    auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
    return ret;
}
}  // namespace

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

bool NpuUtils::IsOomError(aclError ret, int index) {
    if (ret == ACL_ERROR_GE_DEVICE_MEMORY_ALLOCATION_FAILED) {
        int deviceId = 0;
        // free devcie cached memory when return value of the first op execution is
        // oom
        if (index == -1) {
            NPU_CHECK_ERROR(aclrtGetDevice(&deviceId));
        }
        AT_ERROR("NPU out of memory. device id: ", deviceId);
    }
    return false;
}

at::Tensor NpuUtils::format_contiguous(const at::Tensor &src){INTERFACE_NOT_IMPL}

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

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type) {
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type, const string &realDataType) {
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    if (!realDataType.empty()) {
        return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
    }
    return acl_dtype;
}

at::Tensor CalcuOpUtil::CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type) {
    return CalcuOpUtil::CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

at::Tensor CalcuOpUtil::CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(aclrtGetDevice(&deviceIndex));
    return cpuPinMemTensor.to(c10::Device(at_npu::key::NativeDeviceType, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

// OpPreparation part

// used to apply output tensor
at::Tensor OpPreparation::apply_tensor(const at::Tensor &src) {
    return at::empty(src.sizes(), src.options());
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor &src, c10::IntArrayRef sizes) {
    return at::empty(sizes, src.options());
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor &src, const c10::TensorOptions &options) {
    return at::empty(src.sizes(), options);
}

at::Tensor OpPreparation::apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src) {
    return at::empty(sizes, src.options());
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format) {
    INTERFACE_NOT_IMPL
    return src;
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format, bool keep_format) {
    INTERFACE_NOT_IMPL;
    return src;
}

at::Tensor OpPreparation::apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format, bool keep_format) {
    INTERFACE_NOT_IMPL;
}

at::Tensor OpPreparation::apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions &options) { INTERFACE_NOT_IMPL; }

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

std::atomic<uint64_t> ExecuteParas::g_pta_correlation_id{0};

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
    void SetEnginePriority() {
        auto stream = c10_npu::getCurrentNPUStream();
        // if (stream.isDataPreprocessStream()) {
        if (0) {
            AddAttr("_performance_prior", true);
            AddAttr<std::string>("_exclude_engines", "AiCore");
        }
    }

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

    void SetDeterministic() { OP_NOT_IMPL }

private:
    string opName;
    AclExecParam execParam;
};  // class OpCommandImpl

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define ASCEND_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define ASCEND_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define ASCEND_LIKELY(expr) (expr)
#define ASCEND_UNLIKELY(expr) (expr)
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define ASCEND_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#elif defined(_MSC_VER)
#define ASCEND_ALWAYS_INLINE __forceinline
#else
#define ASCEND_ALWAYS_INLINE inline
#endif

#define ACL_REQUIRE_OK_OP(expr, opstr)                                                                                                                   \
    do {                                                                                                                                                 \
        if (ASCEND_UNLIKELY((expr) != 0)) {                                                                                                              \
            printf("%s\n", opstr);                                                                                                                       \
            TORCH_CHECK((expr) == 0, __func__, ":", __FILE__, ":", __LINE__, " NPU error,NPU error code is:", expr, "\n", c10_npu::acl::AclGetErrMsg()); \
        }                                                                                                                                                \
    } while (0)

void OpCommandImpl::Run(bool sync, c10::SmallVector<int64_t, N> &sync_index, c10::SmallVector<at::Tensor, N> &outputTensor) {
    NPU_LOGD("Op %s Run.", opName.c_str());
// RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
#if 0
      if (PyGILState_Check()) {
        // we need to release GIL for NPU to compile op.
        Py_BEGIN_ALLOW_THREADS
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
        Py_END_ALLOW_THREADS
      } else {
        ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
      }
#else
    ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
#endif
}

aclError OpCommandImpl::InnerRun(const string &name, AclExecParam &params, bool sync, c10::SmallVector<int64_t, N> &sync_index,
                                 c10::SmallVector<at::Tensor, N> &outputTensor) {
    aclError ret;
    // at_npu::native::NpuUtils::ProfReportMarkData(name);
    auto stream = c10_npu::getCurrentNPUStream();
    auto inputSize = params.inBuffer.size();
    auto outputSize = params.outBuffer.size();
    // open the deterministicAlgorithms config
    SetDeterministic();
    bool reset_flag = false;
#if 0
    if (ForceJitCompileList::GetInstance().Inlist(name) && env::CheckJitDisable()) {
        AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "enable");
        reset_flag = true;
    }
#endif
    int index = 0;
    do {
        if (params.customHandler) {
            ret = params.customHandler();
            if (ret != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                TORCH_CHECK(false, "Custom hand fail!");
            }
            index++;
            continue;
        }
#if 0
        if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
            at_npu::native::aoe::aoe_manager().IsInWhitelist(name)) {
          ret = at_npu::native::AclGenGraphAndDumpForOp(
              name.c_str(),
              inputSize,
              params.inDesc.data(),
              params.inBuffer.data(),
              outputSize,
              params.outDesc.data(),
              params.outBuffer.data(),
              params.attr,
              ACL_ENGINE_SYS,
              at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
              nullptr);
          if (ret != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!");
          }
        }
#endif
        if (!sync) {
            ret = aclopCompileAndExecute(name.c_str(),
                                         inputSize,
                                         params.inDesc.data(),
                                         params.inBuffer.data(),
                                         outputSize,
                                         params.outDesc.data(),
                                         params.outBuffer.data(),
                                         params.attr,
                                         ACL_ENGINE_SYS,
                                         ACL_COMPILE_SYS,
                                         NULL,
                                         stream);
            NPU_CHECK_ERROR(ret);
        } else {
            int64_t dimSize;
            ret = AclopCompileAndExecuteV2(name.c_str(),
                                           inputSize,
                                           const_cast<aclTensorDesc **>(params.inDesc.data()),
                                           const_cast<aclDataBuffer **>(params.inBuffer.data()),
                                           outputSize,
                                           const_cast<aclTensorDesc **>(params.outDesc.data()),
                                           params.outBuffer.data(),
                                           params.attr,
                                           ACL_ENGINE_SYS,
                                           ACL_COMPILE_SYS,
                                           NULL,
                                           stream);
            NPU_CHECK_ERROR(ret);
            for (size_t i = 0; i < sync_index.size(); i++) {
                c10::SmallVector<int64_t, N> real_shape;
                for (int64_t j = 0; j < outputTensor[sync_index[i]].dim(); j++) {
                    NPU_CHECK_ERROR(aclGetTensorDescDimV2(params.outDesc[sync_index[i]], j, &dimSize));
                    real_shape.emplace_back(dimSize);
                }
                outputTensor[sync_index[i]].resize_(real_shape);
            }
        }
        ++index;
    } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    if (reset_flag) {
        AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "disable");
    }
    return ret;
}

std::tuple<aclTensorDesc *, aclDataBuffer *> CovertTensorToAclInput(const at::Tensor &tensor, const string &descName, const string &forceDataType) {
    at::ScalarType scalarDataType = tensor.scalar_type();
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(scalarDataType, forceDataType);
    auto format = CalcuOpUtil::GetTensorNpuFormat(tensor);
    // const auto &npuDesc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);

    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, tensor.sizes(), static_cast<aclFormat>(format)).SetName(descName).Get();

    // if aclDataType != ACL_STRING, we use storageDims to calculate nums and use nums * tensor element size to
    // calculate buffer size. But if aclDataType = ACL_STRING, STRING tensor size = 1 and storageDims = 0, we can not
    // use it to calculate size, we need from storage_sizes_ to calculate STRING element real size.
    AclTensorBufferMaker buffer(tensor, tensor.numel());
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> CovertHostTensorToAclInput(const at::Tensor &tensor, at::ScalarType type, CompileType compileType,
                                                                        const string &forceDataType, const string &descName) {
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(type, forceDataType);
    const auto &dims = tensor.sizes();
    AclTensorDescMaker desc;
    aclFormat format = ACL_FORMAT_ND;
    auto aclDesc = desc.Create(aclDataType, dims, format).SetPlacement(static_cast<aclMemType>(compileType)).SetName(descName).Get();
    AclTensorBufferMaker buffer(tensor, tensor.numel());
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc *, aclDataBuffer *> CovertToAclOutput(const at::Tensor &tensor, const string &forceDataType) {
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type(), forceDataType);
    auto format = CalcuOpUtil::GetTensorNpuFormat(tensor);
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, tensor.sizes(), static_cast<aclFormat>(format)).Get();
    AclTensorBufferMaker aclBuffer(tensor, tensor.numel());
    auto aclBuff = aclBuffer.Get();
    return std::tie(aclDesc, aclBuff);
}

// This class maintain the position of the current
// OpCommandImpl object in vector, the resources in
// the object is
class OpCommandImpls {
public:
    TORCH_NPU_API static OpCommandImpls *GetInstanceByTid(std::thread::id tid);
    TORCH_NPU_API void Push(OpCommandImpl *&ptr);
    TORCH_NPU_API void Pop();

private:
    int32_t offset = -1;
    c10::SmallVector<OpCommandImpl, N> objs;
};  // class OpCommandImpls

static std::unordered_map<std::thread::id, OpCommandImpls> opcommand_impls_map;
static std::mutex map_mutex;
static bool deterministicaclnn_oldstatus = false;

OpCommandImpls *OpCommandImpls::GetInstanceByTid(std::thread::id tid) {
    if (opcommand_impls_map.find(tid) == opcommand_impls_map.end()) {
        OpCommandImpls impl;
        std::lock_guard<std::mutex> lock(map_mutex);
        opcommand_impls_map[tid] = std::move(impl);
    }
    return &opcommand_impls_map[tid];
}

void OpCommandImpls::Push(OpCommandImpl *&ptr) {
    ++offset;
    if (static_cast<int32_t>(objs.size()) <= offset) {
        OpCommandImpl impl;
        objs.emplace_back(std::move(impl));
    }
    TORCH_CHECK(objs.size() > offset, "OpCommand size (", objs.size(), ") is smaller than offset (", offset, ")");
    ptr = &objs[offset];
}

void OpCommandImpls::Pop() {
    TORCH_CHECK(offset >= 0, "OpCommand current offset should not be less than ", offset);
    offset -= 1;
}

OpCommand::OpCommand() {
    aclCmds = OpCommandImpls::GetInstanceByTid(std::this_thread::get_id());

    aclCmds->Push(aclCmd);
    aclCmd->SetCustomHandler(nullptr);
}

OpCommand::~OpCommand() {}

OpCommand &OpCommand::Name(const string &name) {
    aclCmd->SetName(name);
    return *this;
}

void OpCommand::SetCustomHandler(PROC_FUNC func){INTERFACE_NOT_IMPL}

OpCommand &OpCommand::Expect(UnifiedResult unified_result) {
    commonType = unified_result.common_type;
    resultTypeDefined = unified_result.result_type_defined;
    commonShape = unified_result.common_shape;
    return *this;
}

// None Input
OpCommand &OpCommand::Input() {
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
    AclTensorBufferMaker buffer(nullptr, 0);
    aclCmd->AddInput(aclDesc, buffer.Get());
    return *this;
}

// Tensor Input which need contiguous
OpCommand &OpCommand::Input(const at::Tensor &input, const string &descName, const c10::optional<aclFormat> &sensitive_format, const string &realData) {
    // AddTensorInput(Contiguous(input), c10::ScalarType::Undefined, descName, realData);
    //
    std::tuple<aclTensorDesc *, aclDataBuffer *> res = CovertTensorToAclInput(input, descName, realData);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return *this;
}

// IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
OpCommand &OpCommand::Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType, CompileType compileType, const string &realDtype,
                            const string &descName) {
    INTERFACE_NOT_IMPL
    return *this;
}

namespace {
const uint64_t kStringOffset = 16UL;
const std::string kStringDType = "string";
static std::unordered_map<at::ScalarType, std::vector<double>> floating_limits_map {
  {at::ScalarType::Double, {std::numeric_limits<double>::max(), std::numeric_limits<double>::min()}},
  {at::ScalarType::Float, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
  {at::ScalarType::BFloat16, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
  {at::ScalarType::Half, {65504, -65504}}
};
static std::unordered_map<at::ScalarType, std::vector<long>> integral_limits_map {
  {at::ScalarType::Long, {std::numeric_limits<long>::max(), std::numeric_limits<long>::min()}},
  {at::ScalarType::Int, {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}},
  {at::ScalarType::Byte, {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()}},
  {at::ScalarType::Char, {std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()}},
  {at::ScalarType::Short, {std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min()}}
};
}  // namespace

bool ScalarIsInLimits(const c10::Scalar &scalar, at::ScalarType type) {
  bool scalar_flag = false;
  if (at::isFloatingType(type)) {
    auto value = scalar.to<double>();
    scalar_flag = value <= floating_limits_map[type][0] && value >= floating_limits_map[type][1];
  } else if (at::isIntegralType(type)) {
    auto value = scalar.to<long>();
    scalar_flag = value <= integral_limits_map[type][0] && value >= integral_limits_map[type][1];
  }
  return scalar_flag;
}

// Scalar Input, we will do h2d in launch kernel
OpCommand &OpCommand::Input(const c10::Scalar &input, const at::ScalarType type, CompileType compileType) {
    at::ScalarType scalar_type = type;
    if (commonType.has_value()) {
        scalar_type = commonType.value();
    }

    at::Tensor tensor = ScalarIsInLimits(input, scalar_type) ? at::detail::scalar_tensor_static(input, scalar_type, at::kCPU) : at::scalar_to_tensor(input).to(scalar_type);
    std::tuple<aclTensorDesc *, aclDataBuffer *> res = CovertHostTensorToAclInput(tensor, tensor.scalar_type(), compileType, "", "");
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
    return *this;
}

// Tensor Input which no need contiguous
OpCommand &OpCommand::InputWithoutContiguous(const at::Tensor &input, const string &descName, const string &realData) {
    INTERFACE_NOT_IMPL
    return *this;
}

// Output Tensor
OpCommand &OpCommand::Output(at::Tensor &output, const string &descName, const c10::optional<aclFormat> &sensitive_format, const string &realType) {
    auto res = CovertToAclOutput(output, realType);
    aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
    outputTensor.emplace_back(output);
    return *this;
}

// Run a single op
void OpCommand::Run() {
    aclCmd->SetEnginePriority();
    const string &op_name = aclCmd->GetName();
    bool sync = true;
    c10::SmallVector<int64_t, N> sync_index;
    aclCmd->Run(sync, sync_index, outputTensor);
    if (sync) {
        Sync();
    }
    aclCmd->releaseSource();
    aclCmds->Pop();
}

OpCommand &OpCommand::Sync(c10::SmallVector<int64_t, N> &index) {
    INTERFACE_NOT_IMPL
    Sync();
    return *this;
}

OpCommand &OpCommand::Sync() {
    auto stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtSynchronizeStream(stream));
    return *this;
}

void npu_fast_reshape_(at::Tensor &tensor) {
    /**
      [NOTE] For some reshape cases such as view, unsqueeze, squeeze, flatten,
      storages of them remain unchanged. So we can refresh reshape tensor's metadata
      to obtain matched tensor.
      */

    // restriction 1
    if (!tensor.is_contiguous()) {
        return;
    }
    // restriction 2
    if (!FormatHelper::IsBaseFormatType(tensor)) {
        return;
    }
#if 0
  // restriction 3: reshape case without any numels change
  if ((tensor.numel() != StorageDescHelper::GetMemorySize(tensor)) ||
      StorageDescHelper::MetaDataAreMatch(&tensor)) {
    return;
  }

  // refresh matadata to input tensor
  StorageDescHelper::ReflushDescBySelf(tensor);
  auto base_format = InferFormat::GuessBaseFormat(tensor.sizes());
  NPUNativeFunctions::npu_format_cast_(tensor, base_format);
#else
    INTERFACE_NOT_IMPL
#endif
}

}  // namespace native
}  // namespace at_npu

thread_local diopiContextHandle_t context = nullptr;

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

int current_device() {
    int devId_ = 0;
    ::aclrtGetDevice(&devId_);
    return devId_;
}

C10_NPU_API NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) {
    if (device_index == -1) {
        device_index = current_device();
    }
    TORCH_CHECK(context);
    diopiStreamHandle_t stream_handle;
    diopiGetStream(context, &stream_handle);
    c10::Device device(c10::DeviceType::XPU, device_index);
    c10::Stream atStream(c10::Stream::Default::DEFAULT, device);
    aclrtStream aclStream = reinterpret_cast<aclrtStream>(stream_handle);
    TORCH_CHECK(aclStream);
    return NPUStream(NPUStream::Unchecked::UNCHECKED, atStream, aclStream);
}

}  // namespace c10_npu
