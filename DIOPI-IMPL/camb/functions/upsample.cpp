/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
struct DescData {
    int dim;
    uint64_t total_num;
    uint64_t total_size;
    int dims[8];
};

}  // namespace

extern "C" diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_CHECK(inputTensor.dim() == 4 || inputTensor.dim() == 3, "Camb only supports UpsampleNearest 2d.")
    DIOPI_CHECK(inputTensor.isContiguous(), "inputTensor should be contiguous");
    DIOPI_CHECK(outputTensor.isContiguous(), "inputTensor should be contiguous");

    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    /* DescData *input_desc_ptr = (DescData *)inputDesc.get();
    DescData *output_desc_ptr = (DescData *)outputDesc.get();

    std::cout << "input_desc_ptr shape:" << input_desc_ptr->dims[0] << " " << input_desc_ptr->dims[1] << " " << input_desc_ptr->dims[2] << " "
              << input_desc_ptr->dims[3] << " " << input_desc_ptr->dims[4] << " " << input_desc_ptr->dims[5] << " " << input_desc_ptr->dims[6] << " "
              << input_desc_ptr->dims[7] << std::endl;
    std::cout << "output_desc_ptr shape:" << output_desc_ptr->dims[0] << " " << output_desc_ptr->dims[1] << " " << output_desc_ptr->dims[2] << " "
              << output_desc_ptr->dims[3] << " " << output_desc_ptr->dims[4] << " " << output_desc_ptr->dims[5] << " " << output_desc_ptr->dims[6] << " "
              << output_desc_ptr->dims[7] << std::endl; */

    CnnlInterpDescriptor interpDesc;

    /* float scale_h = static_cast<float>(outputTensor.shape()[2]) / static_cast<float>(inputTensor.shape()[2]);
    float scale_w = static_cast<float>(outputTensor.shape()[3]) / static_cast<float>(inputTensor.shape()[3]); */

    // std::vector<float> scales = {scale_h, scale_w};
    std::vector<float> scales;

    DIOPI_CALL(interpDesc.set(inputDesc.get(), CNNL_INTERP_NEAREST, CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3, scales.data()));

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlInterp_v3(handle, interpDesc.get(), inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data()));

    printDevData(ctx, inputTensor, "inputTensor");
    printDevData(ctx, outputTensor, "outputTensor");

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
