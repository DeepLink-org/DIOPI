/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <set>

#include "../common/acloprunner.hpp"
// #include "../common/debug.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiSin(ctx, input, input);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    if (useAclnn()) {
        std::cout << "\n================begin aclnnSinAdaptor====================" << std::endl;
        // AscendTensor inAT(input);
        // printContiguousTensor(ctx, inAT, "sinin1111111");
        // aclnnSinAdaptor(ctx, input, out);
        // AscendTensor outAT1(out);
        // printContiguousTensor(ctx, outAT1, "sinOut11111");
        // std::cout << "================finish aclnnSinAdaptor====================" << std::endl;
        
        // std::cout << "DEBUG DEBUG CASE0000000(aclTensor, aclTensor):" << std::endl;
        // std::cout << "DEBUG call aclnnSin." << std::endl;
        // printContiguousTensor(ctx, inAT, "sinin");


        // aclTensor* self00 = nullptr;
        // aclTensor* out00 = nullptr;
        // createAclTensor1(input, &self00);
        // createAclTensor1(out, &out00);
        // std::cout << "DEBUG try sin" << std::endl;
        // aclnnAdaptor("aclnnSin", ctx, self00, out00);
        AclTensor inAcl(input), outAcl(out);
        if (!inAcl.defined() || inAcl.numel() == 0) {
            std::cout << "no value, return cos." << std::endl;
            return diopiSuccess;
        }
        aclnnAdaptor("aclnnCos", ctx, inAcl, outAcl);
        // AscendTensor outAT00(out);
        // printContiguousTensor(ctx, outAT00, "outAT00###################################################");

        // std::cout << "DEBUG finish sin000000000000000000000000000" << std::endl;
        
        // std::cout << "DEBUG DEBUG CASE1(AclTensor, aclTensor):" << std::endl;
        // std::cout << "DEBUG call aclnnSin." << std::endl;
        // printContiguousTensor(ctx, inAT, "sinin");


        // aclTensor* out11 = nullptr;
        // AclTensor self11(input);
        // createAclTensor1(out, &out11);
        // std::cout << "DEBUG try sin" << std::endl;
        // aclnnAdaptor("aclnnSin", ctx, self11, out11);
        // AscendTensor outAT11(out);
        // printContiguousTensor(ctx, outAT11, "outAT11###################################################");

        // std::cout << "DEBUG finish sin1111111111111111111111111111" << std::endl;

        
        // std::cout << "DEBUG DEBUG CASE2(aclTensor, AclTensor):" << std::endl;
        // std::cout << "DEBUG call aclnnSin." << std::endl;
        // printContiguousTensor(ctx, inAT, "sinin");


        // aclTensor* self22 = nullptr;
        // createAclTensor1(input, &self22);
        // AclTensor out22(out);
        // std::cout << "DEBUG try sin" << std::endl;
        // aclnnAdaptor("aclnnSin", ctx, self22, out22);
        // AscendTensor outAT22(out);
        // printContiguousTensor(ctx, outAT22, "outAT22###################################################");

        // std::cout << "DEBUG finish sin222222222222222222222222" << std::endl;

        // TODO 排查为什么自动转换类型会导致结果失败。
        // AclTensor inAcl(input), outAcl(out);
        // if (!inAcl.defined() || inAcl.numel() == 0) {
        //     std::cout << "no value, return sin." << std::endl;
        //     return diopiSuccess;
        // }
        // aclnnAdaptor("aclnnSin", ctx, inAcl, outAcl);
        // outAcl.print();
        // AscendTensor outAT33(out);
        // printContiguousTensor(ctx, outAT33, "outAT33###################################################");
        // std::cout << std::endl;
    } else {
        AscendTensor in(input);
        if (0 == in.numel()) {
            return diopiSuccess;
        }

        std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64, diopi_dtype_complex64, diopi_dtype_complex128};

        // only support: float16, float32, int32, int64, double, complex64, complex128.
        if (typeSet.find(in.dtype()) == typeSet.end()) {
            AscendTensor inputA, outA, inputTmp(input), outTmp(out);
            makeTensorLike(ctx, outA, in, diopi_dtype_float32);
            makeTensorLike(ctx, inputA, in, diopi_dtype_float32);
            castTensor(ctx, inputTmp, inputA);
            AclOpRunner<1, 1>("Sin", ctx).addInput(inputA).addOutput(outA).run();
            diopiCastDtype(ctx, out, static_cast<diopiConstTensorHandle_t>(outA));
        } else {
            AclOpRunner<1, 1>("Sin", ctx).addInput(input).addOutput(out).run();
        }
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
