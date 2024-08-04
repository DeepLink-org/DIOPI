/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <ostream>

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

static void printVec(std::vector<int64_t> vec, std::string msg = "") {
    if (msg != "") {
        std::cout << msg << ": ";
    }
    std::cout << "[ ";
    for (auto i : vec) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

static std::vector<AscendTensor> castIntIndicesToLongIndices(diopiContextHandle_t ctx, std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> result;
    for (auto& t : indices) {
        if (!t.defined()) {
            result.push_back(AscendTensor(nullptr));
            continue;
        }
        if (t.dtype() == diopi_dtype_int32) {
            diopiTensorHandle_t indexHandle = nullptr;
            auto shape = t.shape();
            diopiSize_t size = vectorToDiopiSize(shape);
            diopiRequireTensor(ctx, &indexHandle, &size, nullptr, diopi_dtype_int64, diopi_device);
            DIOPI_ASCEND_CALL_ACLNN(aclnnCast, ctx, t, diopi_dtype_int64, indexHandle);
            result.push_back(AscendTensor(indexHandle));
        } else {
            if (t.device() == diopi_host) {
                result.push_back(AscendTensor(hostToDevice(ctx, t.tensorHandle())));
            } else {
                result.emplace_back(t);
            }
        }
    }
    return result;
}

static void checkIndexTensorTypes(const std::vector<AscendTensor>& indices) {
    for (const auto& t : indices) {
        if (t.defined()) {
            diopiDtype_t type = t.dtype();
            ASCEND_CHECK_ABORT(type == diopi_dtype_int64 || type == diopi_dtype_bool || type == diopi_dtype_uint8,
                               "tensors used as indices must be long, byte or bool tensors");
        }
    }
}

static AscendTensor nonZeroTensor(diopiContextHandle_t ctx, const AscendTensor& self) {
    int64_t numELem = self.numel() * self.dim();
    std::vector<int64_t> nShape{self.numel(), self.dim()};
    std::vector<int64_t> nStride(nShape.size(), 1);
    for (int64_t i = nShape.size() - 2; i >= 0; i--) {
        nStride[i] = nStride[i + 1] * nShape[i + 1];
    }

    diopiTensorHandle_t nzBuff = nullptr;
    diopiSize_t nzBuffSize = vectorToDiopiSize(nShape);
    diopiRequireTensor(ctx, &nzBuff, &nzBuffSize, nullptr, diopi_dtype_int64, diopi_device);
    AscendTensor nzTensor(nzBuff);

    auto aclNZTensor = ::aclCreateTensor(
        nShape.data(), nShape.size(), aclDataType::ACL_INT64, nStride.data(), 0, aclFormat::ACL_FORMAT_ND, &numELem, 1, const_cast<void*>(nzTensor.data()));
    DIOPI_ASCEND_CALL_ACLNN(aclnnNonzero, ctx, self, aclNZTensor);

    int64_t* vDims = nullptr;
    uint64_t vDimsNum = 0;
    auto ret = aclGetViewShape(aclNZTensor, &vDims, &vDimsNum);
    ASCEND_CHECK_ABORT(ret == 0, "NonZero aclGetViewShape failed.");

    std::vector<int64_t> nzShape(vDims, vDims + vDimsNum);
    nzTensor = nzTensor.resize(nzShape);

    delete vDims;
    vDims = nullptr;

    diopiTensorHandle_t nzTrans = nullptr;
    std::vector<int64_t> nzTransShape{nzShape[1], nzShape[0]};
    diopiSize_t nzTransSize = vectorToDiopiSize(nzTransShape);
    diopiRequireTensor(ctx, &nzTrans, &nzTransSize, nullptr, diopi_dtype_int64, diopi_device);
    std::vector<int64_t> transDims{1, 0};
    diopiSize_t permuteDims = vectorToDiopiSize(transDims);
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, nzTensor, permuteDims, nzTrans);

    return AscendTensor(nzTrans);
}

static std::vector<AscendTensor> expandIndicesTensors(diopiContextHandle_t ctx, const AscendTensor& self, const std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> result;
    for (auto& t : indices) {
        if (!t.defined()) {
            result.push_back(t);
        } else {
            if (t.dtype() == diopi_dtype_uint8 || t.dtype() == diopi_dtype_bool) {
                ASCEND_CHECK(t.dtype() == diopi_dtype_uint8,
                             "indexing with dtype torch.uint8 is now deprecated,"
                             " please use a dtype torch.bool instead.");
                for (uint64_t j = 0; j < static_cast<uint64_t>(t.dim()); j++) {
                    uint64_t srcIdx = result.size() + j;
                    ASCEND_CHECK_ABORT(t.shape(j) == self.shape(srcIdx),
                                       "The shape of the mask  %ld at index  %ld does not match the shape of the indexed tensor %ld at index %ld",
                                       t.dim(),
                                       j,
                                       self.dim(),
                                       srcIdx);
                }
                AscendTensor non = nonZeroTensor(ctx, t);
                for (int64_t j = 0; j < t.dim(); j++) {
                    result.push_back(non.select(0, j));
                }
            } else {
                result.push_back(t);
            }
        }
    }
    return result;
}

static AscendTensor emptyAscendTensor(const AscendTensor& self, std::vector<int64_t> shape) {
    diopiTensorHandle_t empty = nullptr;
    diopiSize_t size = vectorToDiopiSize(shape);

    return AscendTensor(empty);
}

static aclTensor* createEmptyAclTensor() {
    std::vector<int64_t> nShape{0};
    std::vector<int64_t> nStride{1};
    int64_t storageSize = 0;
    void* storage = nullptr;

    return ::aclCreateTensor(nShape.data(), nShape.size(), aclDataType::ACL_FLOAT16, nStride.data(), 0, aclFormat::ACL_FORMAT_ND, &storageSize, 0, storage);
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    AscendTensor inputAt(input);
    std::vector<AscendTensor> indicesOrigin(nums);
    for (int64_t i = 0; i < nums; i++) {
        if (indices[i] != nullptr) {
            indicesOrigin[i] = AscendTensor(indices[i]);
        }
    }

    // indices on Device: dipu_device
    // nullptr tensor to AscendTensor(nullptr)
    std::vector<AscendTensor> indicesList = castIntIndicesToLongIndices(ctx, indicesOrigin);

    // check index tensor types
    checkIndexTensorTypes(indicesList);

    // expand tensors
    auto indicesExpanded = expandIndicesTensors(ctx, inputAt, indicesList);

    //
    // correct until then
    // std::vector<AscendTensor> allDefinedIndices;
    // for (auto& it : indicesExpanded) {
    //     if (it.defined()) {
    //         allDefinedIndices.push_back(it);
    //     } else {
    //         allDefinedIndices.push_back(AscendTensor());
    //     }
    // }

    std::vector<aclTensor*> allDefinedIndices;
    auto emptyTensor = createEmptyAclTensor();
    for (const auto& idx : indicesExpanded) {
        if (idx.defined()) {
            allDefinedIndices.push_back(aclnn_adaptor::createAclTensorFromAscendTensor(idx));
        } else {
            allDefinedIndices.push_back(emptyTensor);
        }
    }

    // for (auto& t : indicesExpanded) {
    //     printContiguousTensor(ctx, t, "");
    // }

    // output
    std::vector<int64_t> outShape{34, 2, 6, 197};
    diopiSize_t outSize = vectorToDiopiSize(outShape);
    diopiRequireTensor(ctx, out, &outSize, nullptr, inputAt.dtype(), diopi_device);

    DIOPI_ASCEND_CALL_ACLNN(aclnnIndex, ctx, inputAt, allDefinedIndices, *out);

    //         BEGIN_CALL_ACL_OP(input);
    // torch::List<c10::optional<at::Tensor>> indicesAtList;
    // indicesAtList.reserve(nums);
    // for (int i = 0; i < nums; ++i) {
    //     indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    // }

    // auto indicesCast = impl::aten::castIntIndicesToLongIndices(indicesAtList);
    // at::Tensor outAt = op_api::index(inputAt, indicesCast);
    // impl::aten::buildDiopiTensor(ctx, outAt, out);
    // END_CALL_ACL_OP();
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    // BEGIN_CALL_ACL_OP(gradInput, zerosLikeInput, gradOutput);
    // torch::List<c10::optional<at::Tensor>> indicesAtList;
    // indicesAtList.reserve(nums);
    // for (int i = 0; i < nums; ++i) {
    //     indicesAtList.emplace_back(impl::aten::buildATen(indices[i]));
    // }

    // auto indicesCast = impl::aten::castIntIndicesToLongIndices(indicesAtList);
    // op_api::_index_put_impl_(zerosLikeInputAt, indicesCast, gradOutputAt, true, false);
    // gradInputAt.copy_(zerosLikeInputAt);
    // END_CALL_ACL_OP();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
