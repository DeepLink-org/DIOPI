#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"

extern "C" {

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    auto stream = impl::camb::getStream(ctx);
    auto trInput = impl::camb::makeTensor(input);

    CnnlResourceGuard<cnnlHandle_t, cnnlCreate, cnnlDestroy> CnnlHandle;
    cnnlHandle_t handle = CnnlHandle.get();
    DIOPI_CALLCNNL(cnnlSetQueue(handle, stream));

    CnnlResourceGuard<cnnlTensorDescriptor_t, cnnlCreateTensorDescriptor, cnnlDestroyTensorDescriptor> CnnlDesc;
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(convertType(&dtype, trInput.dtype()));
    cnnlTensorDescriptor_t desc = CnnlDesc.get();

    diopiSize_t shape = trInput.shape();
    int dimNb = shape.len;
    std::vector<int> dimStrides(dimNb, 1);
    std::vector<int> dimSize(dimNb);
    diopiSize_t stride = trInput.stride();

    if (dimNb == 0) {
        dimNb = 1;
        dimSize.push_back(1);
        dimStrides.push_back(1);
    } else {
        for (int i = 0; i < dimNb; ++i) {
            dimSize[i] = shape.data[i];
        }
        if (dimNb > 0) {
            for (int i = 0; i < dimNb; ++i) {
                dimStrides[i] = stride.data[i];
            }
        }
    }

    float val;
    if (value->stype <= 7) {
        val = value->ival;
    } else {
        val = value->fval;
    }

    DIOPI_CALLCNNL(cnnlSetTensorDescriptorEx(desc, layout, dtype, dimNb, dimSize.data(), dimStrides.data()));
    DIOPI_CALLCNNL(cnnlFill(handle, val, desc, trInput.data()));
    return diopiSuccess;
}

}  // extern "C"
