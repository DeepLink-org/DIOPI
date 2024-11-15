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

namespace indexProcess {
std::vector<AscendTensor> castIntIndicesToLongIndices(diopiContextHandle_t ctx, std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> result;
    for (auto& t : indices) {
        if (!t.defined()) {
            result.emplace_back(nullptr);
            continue;
        }
        if (t.dtype() == diopi_dtype_int32) {
            diopiTensorHandle_t indexHandle = nullptr;
            auto shape = t.shape();
            diopiSize_t size = vectorToDiopiSize(shape);
            diopiRequireTensor(ctx, &indexHandle, &size, nullptr, diopi_dtype_int64, diopi_device);
            DIOPI_ASCEND_CALL_ACLNN(aclnnCast, ctx, t, diopi_dtype_int64, indexHandle);
            result.emplace_back(indexHandle);
        } else {
            if (t.device() == diopi_host) {
                result.emplace_back(hostToDevice(ctx, t.tensorHandle()));
            } else {
                result.emplace_back(t);
            }
        }
    }
    return result;
}

void checkIndexTensorTypes(const std::vector<AscendTensor>& indices) {
    for (const auto& t : indices) {
        if (t.defined()) {
            diopiDtype_t type = t.dtype();
            ASCEND_CHECK_ABORT(type == diopi_dtype_int64 || type == diopi_dtype_bool || type == diopi_dtype_uint8,
                               "tensors used as indices must be long, byte or bool tensors");
        }
    }
}

AscendTensor nonZeroTensor(diopiContextHandle_t ctx, const AscendTensor& self) {
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
    DIOPI_ASCEND_CALL_ACLNN_SYNC(aclnnNonzero, ctx, self, aclNZTensor);

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

std::vector<AscendTensor> expandIndicesTensors(diopiContextHandle_t ctx, const AscendTensor& self, const std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> result;
    for (auto& t : indices) {
        if (!t.defined()) {
            result.push_back(t);
        } else {
            if (t.dtype() == diopi_dtype_uint8 || t.dtype() == diopi_dtype_bool) {
                ASCEND_CHECK(t.dtype() == diopi_dtype_bool,
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

                auto shape = non.shape();
                shape[0] = 1;
                diopiSize_t size = vectorToDiopiSize(shape);
                std::vector<diopiTensorHandle_t> nons;

                for (int i = 0; i < non.shape(0); i++) {
                    diopiTensorHandle_t tmp = nullptr;
                    diopiRequireTensor(ctx, &tmp, &size, nullptr, diopi_dtype_int64, diopi_device);
                    nons.push_back(tmp);
                }
                std::vector<int64_t> splitSize(non.shape(0), 1);
                diopiSize_t splitSizeDiopi = vectorToDiopiSize(splitSize);
                DIOPI_ASCEND_CALL_ACLNN(aclnnSplitWithSize, ctx, non, splitSizeDiopi, 0, nons);
                for (const auto nj : nons) {
                    AscendTensor njTensor(nj);
                    result.push_back(njTensor.squeeze(0));
                }
            } else {
                result.push_back(t);
            }
        }
    }
    return result;
}

aclTensor* createEmptyAclTensor() {
    std::vector<int64_t> nShape{0};
    std::vector<int64_t> nStride{1};
    int64_t storageSize = 0;
    void* storage = nullptr;

    return ::aclCreateTensor(nShape.data(), nShape.size(), aclDataType::ACL_FLOAT16, nStride.data(), 0, aclFormat::ACL_FORMAT_ND, &storageSize, 0, storage);
}

static std::vector<AscendTensor> indicesExpandedOutplace(std::vector<AscendTensor> indices) {
    bool first = true;
    std::vector<int64_t> sizes;

    for (auto& idx : indices) {
        if (!idx.defined()) {
            continue;
        } else if (first) {
            sizes = idx.shape();
            first = false;
        } else {
            sizes = inferSize(sizes, idx.shape());
        }
    }

    std::vector<AscendTensor> result;
    for (auto& idx : indices) {
        if (!idx.defined() || (idx.shape() == sizes)) {
            result.push_back(idx);
        } else {
            result.push_back(idx.expand(sizes));
        }
    }
    return result;
}

bool hasContiguousSubspace(std::vector<AscendTensor> indices) {  // true if all the non-null tensors are adjacent
    auto isDefined = [](const AscendTensor& tensor) { return tensor.defined(); };
    auto isNull = [](const AscendTensor& tensor) { return !tensor.defined(); };
    auto start = std::find_if(indices.begin(), indices.end(), isDefined);
    auto stop = std::find_if(indices.rbegin(), indices.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

std::tuple<AscendTensor, std::vector<AscendTensor>> transposeToFront(AscendTensor self, std::vector<AscendTensor> indices) {
    std::vector<int64_t> dims;
    std::vector<AscendTensor> transposedIndices;

    dims.reserve(self.dim());
    for (int64_t i = 0; i < self.dim(); i++) {
        if (indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.push_back(indices[i]);
        }
    }

    for (int64_t i = 0; i < self.dim(); i++) {
        if (!indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.push_back(indices[i]);
        }
    }

    return std::make_tuple(self.permute(dims), transposedIndices);
}

std::vector<int64_t> indexReshape(std::vector<AscendTensor> endIndices, int64_t dimsBefore, int64_t dimsAfter) {
    std::vector<int64_t> indexShape;
    for (auto& idx : endIndices) {
        if (idx.defined()) {
            std::vector<int64_t> shape;
            shape.insert(shape.end(), dimsBefore, 1);
            shape.insert(shape.end(), idx.shape().begin(), idx.shape().end());
            shape.insert(shape.end(), dimsAfter, 1);
            if (indexShape.empty()) {
                indexShape = shape;
            } else {
                indexShape = inferSize(indexShape, shape);
            }
        }
    }
    return indexShape;
}

std::vector<int64_t> indexOutputSize(const AscendTensor& self, std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> midIndices = indicesExpandedOutplace(indices);
    while (midIndices.size() < (size_t)self.dim()) {
        midIndices.emplace_back(nullptr);
    }

    AscendTensor src = self;
    std::vector<AscendTensor> endIndices = midIndices;
    if (!hasContiguousSubspace(midIndices)) {
        endIndices.clear();
        std::tie(src, endIndices) = transposeToFront(self, midIndices);
    }

    int64_t dimsBefore = 0;
    int64_t dimsAfter = 0;
    int64_t dimsIndexed = 0;

    std::vector<int64_t> replaceShape;
    std::vector<int64_t> indexedSizes;

    for (size_t dim = 0; dim < endIndices.size(); dim++) {
        if (!endIndices[dim].defined()) {
            if (dimsIndexed == 0) {
                dimsBefore++;
            } else {
                dimsAfter++;
            }
        } else {
            dimsIndexed++;
            replaceShape = endIndices[dim].shape();
            indexedSizes.push_back(src.shape(dim));
        }
    }

    if (std::find(indexedSizes.begin(), indexedSizes.end(), 0) != indexedSizes.end() &&
        std::find(replaceShape.begin(), replaceShape.end(), 0) == replaceShape.end()) {
        ASCEND_CHECK_ABORT(false, "index is out of bounds for dimension with size 0");
    }

    auto selfShape = src.shape();
    int64_t end = dimsBefore + dimsIndexed;
    selfShape.erase(selfShape.begin() + dimsBefore, selfShape.begin() + end);
    selfShape.insert(selfShape.begin() + dimsBefore, replaceShape.begin(), replaceShape.end());

    std::vector<int64_t> indexShape = indexReshape(endIndices, dimsBefore, dimsAfter);
    std::vector<int64_t> outputSize = indexShape;
    if (indexShape != selfShape) {
        outputSize = inferSize(indexShape, selfShape);
    }

    return outputSize;
}

}  // namespace indexProcess

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    AscendTensor inputAt(input);
    std::vector<AscendTensor> indicesOrigin(nums);
    for (int64_t i = 0; i < nums; i++) {
        if (indices[i] != nullptr) {
            indicesOrigin[i] = AscendTensor(indices[i]);
        }
    }

    std::vector<AscendTensor> indicesList = indexProcess::castIntIndicesToLongIndices(ctx, indicesOrigin);
    indexProcess::checkIndexTensorTypes(indicesList);

    auto indicesExpanded = indexProcess::expandIndicesTensors(ctx, inputAt, indicesList);

    std::vector<aclTensor*> allDefinedIndices;
 
    for (const auto& idx : indicesExpanded) {
        if (idx.defined()) {
            allDefinedIndices.push_back(aclnn_adaptor::createAclTensorFromAscendTensor(idx));
        } else {
            auto emptyTensor = createEmptyAclTensor();
            allDefinedIndices.push_back(emptyTensor);
        }
    }

    std::vector<int64_t> outShape = indexProcess::indexOutputSize(inputAt, indicesExpanded);
    diopiSize_t outSize = vectorToDiopiSize(outShape);
    diopiRequireTensor(ctx, out, &outSize, nullptr, inputAt.dtype(), diopi_device);

    DIOPI_ASCEND_CALL_ACLNN(aclnnIndex, ctx, inputAt, allDefinedIndices, *out);
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t zerosLikeInput, diopiConstTensorHandle_t* indices,
                                int64_t nums, diopiConstTensorHandle_t gradOutput) {
    AscendTensor gradInputTensor(gradInput);
    AscendTensor gradOutputTensor(gradOutput);
    if (gradInputTensor.numel() == 0 || gradOutputTensor.numel() == 0) {
        return diopiSuccess;
    }

    std::vector<diopiConstTensorHandle_t> indicesVec;
    indicesVec.reserve(nums);

    for (int i = 0; i < nums; i++) {
        if (indices[i] != nullptr) {
            indicesVec.emplace_back(indices[i]);
        } else {
            int64_t array[1] = {0};
            diopiSize_t size = {array, 1};
            diopiTensorHandle_t emptyTensor = nullptr;
            diopiRequireTensor(ctx, &emptyTensor, &size, nullptr, gradOutputTensor.dtype(), diopi_device);
            indicesVec.emplace_back(emptyTensor);
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceCopy, ctx, gradInput, zerosLikeInput);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, gradInput, indicesVec, gradOutput, true, false);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
