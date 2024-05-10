/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"
#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    std::cout << std::endl;
    diopiDtype_t inputDtype;
    diopiDtype_t tensor1Dtype;
    diopiDtype_t tensor2Dtype;
    diopiDtype_t outDtype;

    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(tensor1, &tensor1Dtype);
    diopiGetTensorDtype(tensor2, &tensor2Dtype);
    diopiGetTensorDtype(out, &outDtype);

    std::cout << "input_dtype = " << inputDtype << std::endl;
    std::cout << "tensor1_dtype = " << tensor1Dtype << std::endl;
    std::cout << "tensor2_dtype = " << tensor2Dtype << std::endl;
    std::cout << "value_dtype = " << value->stype << std::endl;
    std::cout << "out_dtype = " << outDtype << std::endl;

    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (inputNumel != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnAddcmul, ctx, input, tensor1, tensor2, value, out);
    }
    return diopiSuccess;
}

diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    std::cout << std::endl;
    diopiDtype_t inputDtype;
    diopiDtype_t tensor1Dtype;
    diopiDtype_t tensor2Dtype;

    diopiGetTensorDtype(input, &inputDtype);
    diopiGetTensorDtype(tensor1, &tensor1Dtype);
    diopiGetTensorDtype(tensor2, &tensor2Dtype);

    std::cout << "input_dtype = " << inputDtype << std::endl;
    std::cout << "tensor1_dtype = " << tensor1Dtype << std::endl;
    std::cout << "tensor2_dtype = " << tensor2Dtype << std::endl;
    std::cout << "value_dtype = " << value->stype << std::endl;
    
    int64_t inputNumel = 0;
    diopiGetTensorNumel(input, &inputNumel);
    if (inputNumel != 0) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAddcmul, ctx, input, tensor1, tensor2, value);
    }
    
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl