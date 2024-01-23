/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
#define IMPL_ASCEND_ACLNN_ADAPTOR_HPP_

#include <dlfcn.h>

#include <functional>
#include <iostream>

namespace impl {
namespace ascend {
/**
 * 1. 根据函数名称，获取函数地址
 * 2. 依据函数参数，构造函数定义
 * 3. 调用函数
 */

#define ASCEND_CHECK(condition, ...)                                  \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0);

#define __FILENAME__ __FILE__

#define ASCEND_LOGE(fmt, ...)                                                                 \
    aclAppLog(ACL_ERROR, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d \n[PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGW(fmt, ...)                                                                   \
    aclAppLog(ACL_WARNING, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGI(fmt, ...)                                                                \
    aclAppLog(ACL_INFO, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGD(fmt, ...)                                                                 \
    aclAppLog(ACL_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

inline const char* GetOpApiLibName(void) { return "libopapi.so"; }

inline const char* GetCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void* GetOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* GetOpApiLibHandler(const char* libName) {
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void* GetOpApiFuncAddr(const char* apiName) {
    static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
    if (custOpApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
    // static_assert(std::is_same<decltype(f(std::get<I>(t)...)), void>::value,
    //               "call_impl: f(std::get<I>(t)...)");
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>) {
    typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

// ConvertType ConvertType ConvertType ConvertType ConvertType ConvertType ConvertType ConvertType ConvertType
// inline aclTensor* ConvertType(const at::Tensor& at_tensor) {
//   static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
//   if (aclCreateTensor == nullptr) {
//     return nullptr;
//   }

//   if (!at_tensor.defined()) {
//     return nullptr;
//   }
//   TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
//   at::ScalarType scalar_data_type = at_tensor.scalar_type();
//   aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
//   c10::SmallVector<int64_t, 5> storageDims;
//   // if acl_data_type is ACL_STRING, storageDims is empty.
//   if (acl_data_type != ACL_STRING) {
//     storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
//   }

//   const auto dimNum = at_tensor.sizes().size();
//   aclFormat format = ACL_FORMAT_ND;
//   switch (dimNum) {
//     case 3:
//       format = ACL_FORMAT_NCL;
//       break;
//     case 4:
//       format = ACL_FORMAT_NCHW;
//       break;
//     case 5:
//       format = ACL_FORMAT_NCDHW;
//       break;
//     default:
//       format = ACL_FORMAT_ND;
//   }

//   if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
//     c10::Scalar expScalar = at_tensor.item();
//     at::Tensor aclInput = at_npu::native::OpPreparation::copy_scalar_to_device(expScalar, scalar_data_type);
//     return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type, aclInput.strides().data(),
//                            aclInput.storage_offset(), format, storageDims.data(), storageDims.size(),
//                            const_cast<void*>(aclInput.storage().data()));
//   }

//   auto acl_tensor = aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
//                                     at_tensor.strides().data(), at_tensor.storage_offset(), format, storageDims.data(),
//                                     storageDims.size(), const_cast<void*>(at_tensor.storage().data()));
//   return acl_tensor;
// }

// inline aclScalar* ConvertType(const at::Scalar& at_scalar) {
//   static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
//   if (aclCreateScalar == nullptr) {
//     return nullptr;
//   }

//   at::ScalarType scalar_data_type = at_scalar.type();
//   aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
//   aclScalar* acl_scalar = nullptr;
//   switch (scalar_data_type) {
//     case at::ScalarType::Double: {
//       double value = at_scalar.toDouble();
//       acl_scalar = aclCreateScalar(&value, acl_data_type);
//       break;
//     }
//     case at::ScalarType::Long: {
//       int64_t value = at_scalar.toLong();
//       acl_scalar = aclCreateScalar(&value, acl_data_type);
//       break;
//     }
//     case at::ScalarType::Bool: {
//       bool value = at_scalar.toBool();
//       acl_scalar = aclCreateScalar(&value, acl_data_type);
//       break;
//     }
//     case at::ScalarType::ComplexDouble: {
//       auto value = at_scalar.toComplexDouble();
//       acl_scalar = aclCreateScalar(&value, acl_data_type);
//       break;
//     }
//     default:
//       acl_scalar = nullptr;
//       break;
//   }

//   return acl_scalar;
// }

// inline aclIntArray* ConvertType(const at::IntArrayRef& at_array) {
//   static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
//   if (aclCreateIntArray == nullptr) {
//     return nullptr;
//   }
//   auto array = aclCreateIntArray(at_array.data(), at_array.size());
//   return array;
// }

// template <std::size_t N>
// inline aclBoolArray* ConvertType(const std::array<bool, N>& value) {
//   static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
//   if (aclCreateBoolArray == nullptr) {
//     return nullptr;
//   }

//   auto array = aclCreateBoolArray(value.data(), value.size());
//   return array;
// }

// inline aclBoolArray* ConvertType(const at::ArrayRef<bool>& value) {
//   static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
//   if (aclCreateBoolArray == nullptr) {
//     return nullptr;
//   }

//   auto array = aclCreateBoolArray(value.data(), value.size());
//   return array;
// }

// inline aclTensorList* ConvertType(const at::TensorList& at_tensor_list) {
//   static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
//   if (aclCreateTensorList == nullptr) {
//     return nullptr;
//   }

//   std::vector<const aclTensor*> tensor_list(at_tensor_list.size());
//   for (size_t i = 0; i < at_tensor_list.size(); i++) {
//     tensor_list[i] = ConvertType(at_tensor_list[i]);
//   }
//   auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
//   return acl_tensor_list;
// }

// inline aclTensor* ConvertType(const c10::optional<at::Tensor>& opt_tensor) {
//   if (opt_tensor.has_value() && opt_tensor.value().defined()) {
//     return ConvertType(opt_tensor.value());
//   }

//   return nullptr;
// }

// inline aclIntArray* ConvertType(const c10::optional<at::IntArrayRef>& opt_array) {
//   if (opt_array.has_value()) {
//     return ConvertType(opt_array.value());
//   }

//   return nullptr;
// }

// inline aclScalar* ConvertType(const c10::optional<at::Scalar>& opt_scalar) {
//   if (opt_scalar.has_value()) {
//     return ConvertType(opt_scalar.value());
//   }

//   return nullptr;
// }

// inline aclDataType ConvertType(const at::ScalarType scalarType) {
//   return at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType);
// }

template <typename T>
T ConvertType(T value) {
    return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args) {
    return std::make_tuple(ConvertType(args)...);
}

int test(std::string name, diopiContextHandle_t ctx, aclTensor* self, aclTensor* out) {
    
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.1
    // name = "aclnnCos";
    std::string workSpaceName = name + "GetWorkspaceSize";
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workSpaceName.c_str());
    std::cout << "DEBUGDEBUG getWorkspaceSizeFuncAddr =" << getWorkspaceSizeFuncAddr << std::endl;
    ASCEND_CHECK(getWorkspaceSizeFuncAddr != nullptr, "can't get ", name, " workspace function.");

    // 1.2
    uint64_t workspaceSize = 0;
    uint64_t* workspaceSizeAddr = &workspaceSize;
    aclOpExecutor* executor = nullptr;
    aclOpExecutor** executorAddr = &executor;
    auto convertedParams = ConvertTypes(self, out, workspaceSizeAddr, executorAddr);
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr);
    std::cout << "getWorkspaceSizeFunc=" << getWorkspaceSizeFunc << std::endl;

    // 1.3 完成workspace调用逻辑
    auto workspaceStatus = call(getWorkspaceSizeFunc, convertedParams);
    std::cout << "workspace_status=" << workspaceStatus << std::endl;
    ASCEND_CHECK(workspaceStatus != 0, "workspaceStatus= ", workspaceStatus, " , not equal 0.");
    // return 0;

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 2.1 
    static const auto opApiFuncAddr = GetOpApiFuncAddr(name.c_str());
    ASCEND_CHECK(opApiFuncAddr != nullptr, "can't get ", name, " op function.");

    // 2.2
    typedef int(*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
    auto ret = opApiFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCos failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    std::cout << "DEBUGDEBUG DEBUGDEBUG DEBUGDEBUG finish" << std::endl;

    return 0;
}

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
