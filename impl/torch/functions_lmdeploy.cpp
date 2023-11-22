/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>
#include <diopi/functions_lmdeploy.h>

#include <iostream>
#include <vector>

extern "C" {

DIOPI_API diopiError_t diopiFusedSiluFfnInp(diopiContextHandle_t ctx, diopiTensorHandle_t inoutput, diopiConstTensorHandle_t weight1,
                                            diopiConstTensorHandle_t weight2, diopiConstTensorHandle_t weight3, diopiTensorHandle_t workspace,
                                            int64_t* workspace_size, int64_t fusion_level) {
    if (fusion_level >= 0) {
        diopiSize_t shapeinfo;
        diopiGetTensorShape(inoutput, &shapeinfo);
        int64_t token_num = shapeinfo.data[0];
        diopiGetTensorShape(weight1, &shapeinfo);
        int64_t inter_size = shapeinfo.data[1];
        int64_t itemsize = -1;
        diopiGetTensorElemSize(inoutput, &itemsize);
        if (*workspace_size < 0) {
            *workspace_size = 2 * itemsize * token_num * inter_size;
            return diopiSuccess;
        }
        void* dataptr;
        diopiGetTensorData(workspace, &dataptr);
        diopiDevice_t device;
        diopiGetTensorDevice(workspace, &device);
        diopiDtype_t dtype;
        diopiGetTensorDtype(workspace, &dtype);
        std::vector<int64_t> shape(2);
        diopiSize_t newshape{shape.data(), 2};
        shape[0] = token_num;
        shape[1] = inter_size;
        diopiSize_t strideW1{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(dataptr)), -1};
        diopiSize_t strideW3{static_cast<const int64_t*>(reinterpret_cast<int64_t*>(reinterpret_cast<char*>(dataptr) + itemsize * token_num * inter_size)), -1};
        diopiTensorHandle_t matmulW1;
        diopiTensorHandle_t matmulW3;
        diopiRequireTensor(ctx, &matmulW1, &newshape, &strideW1, dtype, device);
        diopiRequireTensor(ctx, &matmulW3, &newshape, &strideW3, dtype, device);

        diopiMm(ctx, matmulW1, inoutput, weight1);
        diopiMm(ctx, matmulW3, inoutput, weight3);
        diopiSiluInp(ctx, matmulW1);
        diopiMulInp(ctx, matmulW1, matmulW3);
        diopiMm(ctx, inoutput, matmulW1, weight2);
        return diopiSuccess;
    }
    return diopiErrorOccurred;
}

}  // extern "C"
