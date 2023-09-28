/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

#include "../ascend_tensor.hpp"

namespace impl{
namespace ascend{
extern "C"{

DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                                    diopiConstTensorHandle_t batch2, double beta, double alpha){
/**
 * @brief Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
 * \f[ out = \beta \times input + \alpha (batch_1 @ batch_2) \f]
 * @param[in] ctx Context environment.
 * @param[in] input the tensor to be added.
 * @param[in] batch1 the first batch of matrices to be multiplied.
 * @param[in] batch2 the second batch of matrices to be multiplied.
 * @param[in] beta multiplier for input.
 * @param[in] alpha multiplier for batch1 and batch2.
 * @param[out] out the output tensor.
 */
//out i =β input i ​+ α (batch1 i @ batch2 i)
    std::cout<<"enter"<<std::endl;

    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);
    std::cout<<"outDtype: "<<outDtype<<std::endl;

    //input types include: float16, float32, float64
    
    //adjust the input's and output's data type
    diopiDtype_t execType = diopi_dtype_float32;
    
    diopiTensorHandle_t inCopy;
    makeTensorLike(ctx, &inCopy, input, execType);
    diopiCastDtype(ctx, inCopy, input);
    AscendTensor as_inCopy(inCopy);

    diopiTensorHandle_t outCopy;
    makeTensorLike(ctx, &outCopy, out, execType);
    diopiCastDtype(ctx, outCopy, out);

    diopiTensorHandle_t batch1Copy;
    
    makeTensorLike(ctx, &batch1Copy, batch1, execType);
    diopiCastDtype(ctx, batch1Copy, batch1);

    diopiTensorHandle_t batch2Copy;
    makeTensorLike(ctx, &batch2Copy, batch2, execType);
    diopiCastDtype(ctx, batch2Copy, batch2);
    
    //get the output of batch1 time batch2 size
    AscendTensor asTensor1 = AscendTensor(batch1);
    AscendTensor asTensor2 = AscendTensor(batch2);
    std::vector<int64_t> batch1_shape = asTensor1.shape();
    std::vector<int64_t> batch2_shape = asTensor2.shape();
    std::vector<int64_t> vectorSize_BatchMatMulTensor = {batch1_shape[0], batch1_shape[1], batch2_shape[2]};

    
    std::cout<<"batch1_shpae: "<<std::endl;
    for(auto i:batch1_shape)std::cout<<i<<" ";
    std::cout<<std::endl;
    std::cout<<"batch2_shpae: "<<std::endl;
    for(auto i:batch2_shape)std::cout<<i<<" ";
    std::cout<<std::endl;

    //Initialize a tensor according to output's size
    diopiSize_t diopiSize_BatchMatMulTensor = vectorToDiopiSize(vectorSize_BatchMatMulTensor);
    AscendTensor as_BatchMatMulTensor;
    makeTensor(ctx, as_BatchMatMulTensor, &diopiSize_BatchMatMulTensor, diopi_dtype_float32, diopiDevice_t::diopi_device);

    std::cout<<"after allocater mem BatchMatMulTensor'shape:"<<std::endl;
    auto as_bmm_size = as_BatchMatMulTensor.shape();
    for(auto i:as_bmm_size)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    //Does batch1/batch2 need to transpose?
    bool isSelfT = false;
    bool isMat2T = false;

    //diopiTensorHandle_t diopiBatchMatMulTensor = const_cast<diopiTensorHandle_t>(AT_BatchMatMulTensor.tensorHandle());

    //do batch1 times batch2 -> BatchMatMulTensor
    AclOpRunner<5, 1>("BatchMatMul", ctx)
        .addInput(batch1Copy)
        .addInput(batch2Copy)
        .addOutput(as_BatchMatMulTensor)
        .setAttr("adj_x1", isSelfT)
        .setAttr("adj_x2", isMat2T)
        .run();

    // auto afterrun_size_BatchMatMulTensor = AT_BatchMatMulTensor.shape();

    // std::cout<<"aften running the result size: "<<std::endl;
    // for(auto i:afterrun_size_BatchMatMulTensor)
    //     std::cout<<i<<" ";
    // std::cout<<std::endl;

    // std::cout<<" init alphaMulTensor and  betaMulTensor"<<std::endl;
    //init memory based on the size of alphaMulTensor and betaMulTensor
    AscendTensor alphaMulTensor ;
    AscendTensor betaMulTensor ;
    makeTensorLike(ctx, alphaMulTensor, as_BatchMatMulTensor, diopi_dtype_float32);
    makeTensorLike(ctx, betaMulTensor, as_BatchMatMulTensor, diopi_dtype_float32);

    // std::cout<<" init alpha_scalar and  beta_scalar"<<std::endl;
    diopiScalar_t alpha_scalar;
    diopiScalar_t beta_scalar;
    alpha_scalar.fval = alpha;
    alpha_scalar.stype = diopi_dtype_float32;
    beta_scalar.fval = beta;
    beta_scalar.stype = diopi_dtype_float32;

    std::cout<<"input's type: "<<as_inCopy.dtype()<<std::endl;
    std::cout<<"alphaMulTensor's type: "<<alphaMulTensor.dtype()<<std::endl;
    std::cout<<"betaMulTensor's type: "<<betaMulTensor.dtype()<<std::endl;
    std::cout<<"as_BatchMatMulTensor's type: "<<as_BatchMatMulTensor.dtype()<<std::endl;
    std::cout<<std::endl;
    auto size_in = as_inCopy.shape();
    std::cout<<"input size: "<<std::endl;
    for(auto i:size_in)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    auto size_alphaMT = alphaMulTensor.shape();
    std::cout<<"alphaMulTensor size: "<<std::endl;
    for(auto i:size_alphaMT)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    auto size_betaMulTensor = betaMulTensor.shape();
    std::cout<<"betaMulTensor size: "<<std::endl;
    for(auto i:size_betaMulTensor)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    auto size_as_BatchMatMulTensor = as_BatchMatMulTensor.shape();
    std::cout<<"as_BatchMatMulTensor size: "<<std::endl;
    for(auto i:size_as_BatchMatMulTensor)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    diopiTensorHandle_t diopiAlphaMulTensor = const_cast<diopiTensorHandle_t>(alphaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiBateMulTensor = const_cast<diopiTensorHandle_t>(betaMulTensor.tensorHandle());
    diopiTensorHandle_t diopi_as_BatchMatMulTensor = const_cast<diopiTensorHandle_t>(as_BatchMatMulTensor.tensorHandle());

    std::cout<<"Bfter mul scalar"<<std::endl;
    //\alpha times BatchMatMulTensor -> alphaMulTensor and \beta times input -> betaMulTensor
    //diopiConstTensorHandle_t diopi_BatchMatMulTensor = AT_BatchMatMulTensor.tensorHandle();
    diopiMulScalar(ctx, diopiAlphaMulTensor, diopi_as_BatchMatMulTensor, &alpha_scalar);
    diopiMulScalar(ctx, diopiBateMulTensor, inCopy, &beta_scalar);

    std::cout<<"Bfter mul add"<<std::endl;
    diopiScalar_t other;
    other.fval = 1;
    other.fval = diopi_dtype_float32;

    // AscendTensor as_outCopy = AscendTensor(outCopy);
    // auto size_outCopy = as_outCopy.shape(); 
    // std::cout<<"size_outCopy size: "<<std::endl;
    // for(auto i:size_outCopy)
    //     std::cout<<i<<" ";
    // std::cout<<std::endl;

    // AscendTensor as_AlphaMulTensor = AscendTensor(diopiAlphaMulTensor);
    // auto size_AlphaMulTensor = as_AlphaMulTensor.shape(); 
    // std::cout<<"size_AlphaMulTensor size: "<<std::endl;
    // for(auto i:size_AlphaMulTensor)
    //     std::cout<<i<<" ";
    // std::cout<<std::endl;
    
    diopiAdd(ctx, outCopy, diopiAlphaMulTensor, diopiBateMulTensor, &other);
    diopiCastDtype(ctx, out, outCopy);

    AscendTensor as_out = AscendTensor(out);
    auto size_as_out = as_out.shape();
    std::cout<<"as_out size: "<<std::endl;
    for(auto i:size_as_out)
        std::cout<<i<<" ";
    std::cout<<std::endl;

    std::cout<<"exit"<<std::endl;

    return diopiSuccess;
}


}
} // namespace ascend
} //namespace impl