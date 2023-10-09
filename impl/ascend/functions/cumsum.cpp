/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiDtype_t inputDataType;
    diopiDtype_t outputDataType;
    diopiGetTensorDtype(input, &inputDataType);
    diopiGetTensorDtype(out, &outputDataType);
    
    if(inputDataType != outputDataType){
        diopiTensorHandle_t inCopy;
        makeTensorLike(ctx, &inCopy, input, outputDataType);
        diopiCastDtype(ctx, inCopy, input);
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(inCopy).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    }else if(inputDataType == diopi_dtype_bool){
        diopiTensorHandle_t inCopy;
        makeTensorLike(ctx, &inCopy, input, diopi_dtype_uint8);
        diopiCastDtype(ctx, inCopy, input);
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(inCopy).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    }else{
        AclOpRunner<2, 1>("Cumsum", ctx).addInput(input).addConstInput(dim, diopi_dtype_int64).addOutput(out).run();
    }
    
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
