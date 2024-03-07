/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace OP_IMPL_NS {

static const int64_t bitNumber = 128;
static const int64_t uInt8BitNumber = 8;

static diopiError_t ascendNpuDropoutOut(at::Tensor& out, at::Tensor& mask, const at::Tensor& input, double p, at::Generator gen) {
    DIOPI_CHECK(GetOpApiFuncAddr("aclnnDropoutGenMask"
                                 "GetWorkspaceSize") != nullptr &&
                    GetOpApiFuncAddr("aclnnDropoutGenMask") != nullptr,
                "[DIOPI][Ascend] aclnnDropoutGenMask not find, check it supported by AscendCL");
    DIOPI_CHECK(GetOpApiFuncAddr("aclnnDropoutDoMask"
                                 "GetWorkspaceSize") != nullptr &&
                    GetOpApiFuncAddr("aclnnDropoutDoMask") != nullptr,
                "[DIOPI][Ascend] aclnnDropoutDoMask not find, check it supported by AscendCL");
    DIOPI_CHECK(GetOpApiFuncAddr("aclnnNeScalar"
                                 "GetWorkspaceSize") != nullptr &&
                    GetOpApiFuncAddr("aclnnNeScalar") != nullptr,
                "[DIOPI][Ascend] aclnnNeScalar not find, check it supported by AscendCL");
    int64_t length = (input.numel() + bitNumber - 1) / bitNumber * bitNumber / uInt8BitNumber;
    at::Tensor maskNpu = at_npu::native::OpPreparation::apply_tensor_without_format({length}, input.options().dtype(at::kByte));
    at::IntArrayRef shapeArray(input.sizes());

    auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
    const uint64_t seed = pair.first;
    const uint64_t offset = pair.second;

    EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, p, seed, offset, maskNpu);
    EXEC_NPU_CMD(aclnnDropoutDoMask, input, maskNpu, p, out);

    at::Scalar ref(0.0);
    at_npu::native::OpPreparation::check_tensor({out}, mask, mask.scalar_type(), out.sizes());
    EXEC_NPU_CMD(aclnnNeScalar, out, ref, mask);

    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train,
                          diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(out, input, mask, generator);

    DIOPI_CHECK(outAt.defined() && outAt.sizes() == inputAt.sizes(), "[DIOPI][Ascend] Check if out defined or out shape equal to input shape");
    DIOPI_CHECK(maskAt.defined(), "[DIOPI][Ascend] Check if mask tensor defined");

    if (p == 0 || train == false) {
        outAt.copy_(inputAt);
        op_api::fill_(maskAt, c10::Scalar(1));
        END_CALL_ACL_OP();
    }
    if (p == 1) {
        op_api::fill_(outAt, c10::Scalar(0));
        op_api::fill_(maskAt, c10::Scalar(0));
        END_CALL_ACL_OP();
    }

    if (inputAt.sizes() != maskAt.sizes()) {
        auto input2d = op_api::ones_like(maskAt, inputAt.scalar_type());
        auto results = op_api::_npu_dropout(input2d, p);
        op_api::mul_out(inputAt, std::get<0>(results), outAt);
        op_api::ne_out(std::get<0>(results), c10::Scalar(0), maskAt);
    } else {
        ascendNpuDropoutOut(outAt, maskAt, inputAt, p, generatorAt);
    }

    END_CALL_ACL_OP();
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    BEGIN_CALL_ACL_OP(input, mask, generator);

    DIOPI_CHECK(maskAt.defined(), "[DIOPI][Ascend] Check if mask tensor defined");

    if (p == 0 || train == false) {
        op_api::fill_(maskAt, c10::Scalar(1));
        END_CALL_ACL_OP();
    }
    if (p == 1) {
        op_api::fill_(inputAt, c10::Scalar(0));
        op_api::fill_(maskAt, c10::Scalar(0));
        END_CALL_ACL_OP();
    }
    if (inputAt.sizes() != maskAt.sizes()) {
        auto input2d = op_api::ones_like(maskAt, inputAt.scalar_type());
        auto results = op_api::_npu_dropout(input2d, p);
        op_api::mul_(inputAt, std::get<0>(results));
        op_api::ne_out(std::get<0>(results), c10::Scalar(0), maskAt);
    } else {
        ascendNpuDropoutOut(inputAt, maskAt, inputAt, p, generatorAt);
    }
    END_CALL_ACL_OP();
}

}  // namespace OP_IMPL_NS
