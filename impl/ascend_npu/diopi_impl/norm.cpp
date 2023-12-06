/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace {
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& normal_out_npu_nocheck(at::Tensor& result, c10::optional<at::Generator> gen) {
    auto gen_default = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen_default->philox_engine_inputs(10);
    const int64_t seed = pair.first;
    const int64_t offset = pair.second;

    at::SmallVector<int64_t, N> key = {seed};
    at::SmallVector<int64_t, N> counter = {0, offset};
    const int32_t alg = 1;

    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessRandomNormalV2")
        .Input(result.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(key, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string) "uint64")
        .Input(counter, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, (string) "uint64")
        .Input(at::Scalar(alg), at::ScalarType::Int)
        .Output(result)
        .Attr("dtype", result.scalar_type())
        .Run();
    return result;
}
}  // namespace

namespace OP_IMPL_NS {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, generator);
    normal_out_npu_nocheck(outAt, c10::make_optional(std::move(generatorAt)));
    END_CALL_ACL_OP();
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(inout, generator);

    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
