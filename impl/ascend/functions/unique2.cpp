// /**
//  * @file
//  * @author DeepLink
//  * @copyright  (c) 2023, DeepLink.
//  */

// #include <cstdint>
// #include <iostream>
// #include <vector>

// #include "../aclnn/adaptor.hpp"

// namespace impl {
// namespace ascend {

// diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
//                          bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
//     bool return_inverse = (indices != nullptr) ? true : false;
//     AscendTensor inputAt(input);

//     std::cout << "diopiUnique" << std::endl;
//     // int64_t local_dim = dim ? (*dim >= 0 ? *dim : *dim + inputAt.dim()) : -1;
//     std::vector<int64_t> inSizeVec = (dim != nullptr) ? std::vector<int64_t>{inputAt.shape(*dim)} : inputAt.shape();
//     diopiSize_t inSize = {inSizeVec.data(), static_cast<int64_t>(inSizeVec.size())};
//     std::cout << "inSizeVec=" << std::endl;
//     for (auto i : inSizeVec) {
//         std::cout << i << std::endl;
//     }
//     std::vector<int64_t> zeroShape = {0};
//     diopiSize_t zeroSize = {zeroShape.data(), static_cast<int64_t>(zeroShape.size())};
//     if (return_counts) {
//         diopiRequireTensor(ctx, counts, &inSize, nullptr, diopi_dtype_int64, diopi_device);
//         if (indices == nullptr) {
//             diopiRequireTensor(ctx, &indices, &inSize, nullptr, diopi_dtype_int64, diopi_device);
//         }
//     } else {
//         diopiRequireTensor(ctx, counts, &zeroSize, nullptr, diopi_dtype_int64, diopi_device);
//         if (indices == nullptr) {
//             diopiRequireTensor(ctx, &indices, &zeroSize, nullptr, diopi_dtype_int64, diopi_device);
//         }
//     }
//     diopiTensorHandle_t outTmp = nullptr;
//     diopiRequireTensor(ctx, &outTmp, &inSize, nullptr, inputAt.dtype(), diopi_device);
//     AscendTensor countsAt(*counts);
//     auto params = ::impl::ascend::aclnn_adaptor::convertParams(input, sorted, return_inverse, return_counts, outTmp, indices, countsAt).params();
    
//     if (dim) {
//         std::cout << "dim=" << *dim << std::endl;
//     } else {
//         std::cout << "all dim" << std::endl;
//     }
//     DIOPI_ASECND_CALL_ACLNN_TYPE_SYNC(aclnnUnique2, ctx, params);
//     std::cout << "diopiUnique finish" << std::endl;

//     // get true outShape by aclGetViewShape
//     int64_t* viewDims = nullptr;
//     uint64_t viewDimNum = 0;
//     using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** viewDims, uint64_t* viewDimsNum);
//     static aclGetViewShapeFunc aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(impl::ascend::aclnn_adaptor::getOpApiFuncAddr("aclGetViewShape"));
//     int ret = aclGetViewShape(std::get<4>(params), &viewDims, &viewDimNum);
//     ASCEND_CHECK_ABORT(ret == 0, "aclGetViewShape failed");
//     diopiSize_t outShape{viewDims, static_cast<int64_t>(viewDimNum)};
//     std::vector<int64_t> outShapeVec(viewDims, viewDims+ viewDimNum);
//     std::cout << "outShapeVec=" << std::endl;
//     for (auto i : outShapeVec) {
//         std::cout << i << std::endl;
//     }
//     // require out tensor from true outShape
//     diopiRequireTensor(ctx, out, &outShape, nullptr, inputAt.dtype(), diopi_device);
//     // copy outTmp to out
//     AscendTensor outAt(*out);
//     AscendTensor outTmpAt(outTmp);
//     outTmpAt.view({outShape.data, outShape.data + outShape.len});

//     std::vector<int64_t> outSizeVec(outShape.data, outShape.data + outShape.len);
//     std::cout << "outSizeVec" << std::endl;
//     for (auto i : outSizeVec) {
//         std::cout << i << " ";
//     }
//     std::cout << "outSizeVec" << std::endl;
//     DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, outAt, outTmpAt);
//     if (viewDims) {
//         delete viewDims;
//         viewDims = nullptr;
//     }
//     return diopiSuccess;
// }

// }  // namespace ascend
// }  // namespace impl
