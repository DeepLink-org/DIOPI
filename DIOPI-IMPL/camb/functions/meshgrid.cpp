/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    for (int i = 0; i < inputsNum; i++) {
        DiopiTensor input_tensor(inputs[i]);
        DiopiTensor out_tensor(outs[i]);

        auto input_dim = input_tensor.shape();
        auto output_dims = out_tensor.shape();

        int tmp_output_dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int tmp_input_dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        int repeat_dim0 = 1;
        int repeat_dim1 = 1;
        for (int j = 0; j < i; j++) {
            repeat_dim0 *= output_dims[j];
        }
        for (int k = i + 1; k < inputsNum; k++) {
            repeat_dim1 *= output_dims[k];
        }
        tmp_output_dims[0] = repeat_dim0 * output_dims[i];
        tmp_output_dims[1] = repeat_dim1;
        tmp_input_dims[0] = output_dims[i];
        tmp_input_dims[1] = 1;

        CnnlTensorDesc input_desc;
        CnnlTensorDesc out_desc;
        std::vector<int> in_dims = {tmp_input_dims[0], tmp_input_dims[1]};
        std::vector<int> out_dims = {tmp_output_dims[0], tmp_output_dims[1]};
        input_desc.set(input_tensor, CNNL_LAYOUT_ARRAY, in_dims);
        out_desc.set(out_tensor, CNNL_LAYOUT_ARRAY, out_dims);

        DIOPI_CALLCNNL(cnnlTile(handle, input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor.data()));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
