/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t broadcast(diopiContextHandle_t ctx, AscendTensor& out, const AscendTensor& input, const std::vector<int64_t>& size) {
    if (!out.defined()) {
        makeTensor(ctx, out, size, input.dtype());
    }
    auto ptr = const_cast<diopiTensorHandle_t>(out.tensorHandle());
    AclOpRunner<2, 1>("BroadcastTo", ctx).addInput(input, diopi_dtype_int32).addConstInput(size).addOutput(ptr).run();
    out = AscendTensor(ptr);
    return diopiSuccess;
}

std::vector<int64_t> inferSize(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    size_t dimsA = shape1.size();
    size_t dimsB = shape2.size();
    size_t ndim = dimsA > dimsB ? dimsA : dimsB;
    std::vector<int64_t> expandedSizes(ndim);

    // Use ptrdiff_t to ensure signed comparison.
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
        ptrdiff_t offset = ndim - 1 - i;
        ptrdiff_t dimA = dimsA - 1 - offset;
        ptrdiff_t dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? shape1[dimA] : 1;
        auto sizeB = (dimB >= 0) ? shape2[dimB] : 1;

        // 1s map to the other size (even 0).
        expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
    }

    return expandedSizes;
}

std::vector<int64_t> inferSize(std::vector<int64_t>& shape, int64_t numel) {
    auto res = shape;
    int64_t newsize = 1;
    // N.B. this is an index, not a sym dim!
    int64_t* inferDim = nullptr;
    for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
        if (shape[dim] == -1) {
            if (inferDim) {
                throw std::runtime_error("only one dimension can be inferred");
            }
            *inferDim = dim;
        } else if (shape[dim] >= 0) {
            newsize *= shape[dim];
        }
    }

    if (numel == newsize || (inferDim && newsize > 0 && numel % newsize == 0)) {
        if (inferDim) {
            res[*inferDim] = numel / newsize;
        }
    }
    return res;
}

bool hasContiguousSubspace(const std::vector<AscendTensor>& tl) {
    // true if all the non-null tensors are adjacent
    auto isDefined = [](const AscendTensor& tensor) { return tensor.defined(); };
    auto isNull = [](const AscendTensor& tensor) { return !tensor.defined(); };
    auto start = std::find_if(tl.begin(), tl.end(), isDefined);
    auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

std::vector<int64_t> indexOutputSize(diopiContextHandle_t ctx, const AscendTensor& self, const std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> newIndices;
    for (const auto& index : indices) {
        if (index.defined() && index.dtype() == diopi_dtype_bool) {
            // ASCEND_CHECK_ABORT(false, "unsupport `index`, cause we need `select` first.");
            // Replace with nonzeros
            AscendTensor nonzero;
            makeTensorLike(ctx, nonzero, index);
            auto outPtr = const_cast<diopiTensorHandle_t>(nonzero.tensorHandle());
            diopiNonzero(ctx, &outPtr, index.tensorHandle());
            nonzero = AscendTensor(outPtr);
            for (int64_t j = 0; j < index.dim(); j++) {
                AscendTensor newIndex;
                makeTensorLike(ctx, newIndex, nonzero);
                diopiTensorHandle_t indexJ;
                auto scalarJ = constructDiopiScalarT(diopi_dtype_int64, 1);
                makeTensorFromScalar(ctx, &scalarJ, &indexJ);
                diopiIndexSelect(ctx, const_cast<diopiTensorHandle_t>(newIndex.tensorHandle()), nonzero.tensorHandle(), 1, indexJ);
                newIndices.emplace_back(newIndex);
            }
        } else {
            newIndices.emplace_back(index);
        }
    }

    std::vector<int64_t> inferShape;
    for (const auto& newIndice : newIndices) {
        if (!newIndice.defined()) {
            continue;
        } else if (inferShape.empty()) {
            inferShape = newIndice.shape();
        } else {
            inferShape = inferSize(inferShape, newIndice.shape());
        }
    }

    std::vector<AscendTensor> midIndices(newIndices.size());
    for (size_t i = 0; i < newIndices.size(); ++i) {
        if (!newIndices[i].defined()) {
            continue;
        } else if (newIndices[i].shape() == (inferShape)) {
            midIndices[i] = newIndices[i];
        } else {
            AscendTensor out;
            makeTensor(ctx, out, inferShape, newIndices[i].dtype());
            broadcast(ctx, out, newIndices[i], inferShape);
            midIndices[i] = out;
        }
    }

    while (midIndices.size() < (size_t)self.dim()) {
        midIndices.emplace_back();
    }
    AscendTensor src = self;
    std::vector<AscendTensor> endIndices = midIndices;
    if (!hasContiguousSubspace(midIndices)) {
        endIndices.clear();
        std::vector<int64_t> dims;
        dims.reserve(self.dim());
        for (int64_t i = 0; i < self.dim(); i++) {
            if (midIndices[i].defined()) {
                dims.push_back(i);
                endIndices.emplace_back(midIndices[i]);
            }
        }
        for (int64_t i = 0; i < self.dim(); i++) {
            if (!midIndices[i].defined()) {
                dims.push_back(i);
                endIndices.emplace_back();
            }
        }
        auto inPtr = const_cast<diopiTensorHandle_t>(self.tensorHandle());
        auto outPtr = const_cast<diopiTensorHandle_t>(src.tensorHandle());
        diopiSize_t dimVec = vectorToDiopiSize(dims);
        diopiPermute(ctx, outPtr, inPtr, dimVec);
        src = AscendTensor(outPtr);
    }

    int64_t dimsBefore = 0, dimsAfter = 0, dimsIndexed = 0;
    std::vector<int64_t> replacementShape;
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
            replacementShape = endIndices[dim].shape();
            indexedSizes.push_back(src.shape(dim));
        }
    }
    auto selfShape = std::vector<int64_t>(src.shape());
    int64_t end = dimsBefore + dimsIndexed;
    selfShape.erase(selfShape.begin() + dimsBefore, selfShape.begin() + end);
    selfShape.insert(selfShape.begin() + dimsBefore, replacementShape.begin(), replacementShape.end());

    std::vector<int64_t> indexShape;
    for (auto& index : endIndices) {
        if (index.defined()) {
            std::vector<int64_t> shape;
            std::vector<int64_t> before(dimsBefore, 1);
            shape.insert(shape.end(), before.begin(), before.end());

            for (int i = 0; i < index.shape().size(); ++i) {
                shape.push_back(index.shape(i));
            }
            std::vector<int64_t> after(dimsAfter, 1);
            shape.insert(shape.end(), after.begin(), after.end());
            if (indexShape.empty()) {
                indexShape = shape;
            } else if (indexShape != shape) {
                indexShape = inferSize(indexShape, shape);
            }
        }
    }

    std::vector<int64_t> outputSize = indexShape;
    if (indexShape != selfShape) {
        outputSize = inferSize(indexShape, selfShape);
    }

    return outputSize;
}

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    std::vector<int64_t> dimVec({dim});
    diopiSize_t dimInput = vectorToDiopiSize(dimVec);
    AclOpRunner<3, 1>("GatherV2", ctx).addInput(input).addInput(index).addConstInput(dimInput).setAttr<int64_t>("batch_dims", 0).addOutput(out).run();
    return diopiSuccess;
}

std::vector<AscendTensor> filterDefinedTensors(const std::vector<AscendTensor>& indices) {
    std::vector<AscendTensor> result;
    for (auto& index : indices) {
        if (!index.numel()) {
            result.emplace_back();
        } else {
            result.emplace_back(index);
        }
    }
    return result;
}

std::vector<AscendTensor> broadcastTensors(diopiContextHandle_t ctx, const std::vector<AscendTensor>& tensors) {
    // Broadcast a list of Tensors, ignoring undefined (null) tensors.
    bool first = true;
    std::vector<int64_t> sizes;
    for (const auto& tensor : tensors) {
        if (!tensor.defined()) {
            continue;
        } else if (first) {
            // The initial value of sizes is the first defined tensor's shape.
            sizes = tensor.shape();
            first = false;
        } else {
            sizes = inferSize(sizes, tensor.shape());
        }
    }

    std::vector<AscendTensor> result(tensors.size());
    for (int i = 0; i < tensors.size(); ++i) {
        if (!tensors[i].defined()) {
            continue;
        } else if (tensors[i].shape() == sizes) {
            result[i] = tensors[i];
        } else {
            broadcast(ctx, result[i], tensors[i], sizes);
        }
    }
    return result;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    std::vector<AscendTensor> indicesVec;
    for (int i = 0; i < nums; ++i) {
        AscendTensor tmp = AscendTensor(indices[i]);
        indicesVec.emplace_back(tmp);
    }
    // indicesVec = filterDefinedTensors(indicesVec);
    indicesVec = broadcastTensors(ctx, indicesVec);
    std::vector<int64_t> masks;
    std::vector<AscendTensor> realIndices;
    for (auto& i : indicesVec) {
        if (i.defined()) {
            masks.emplace_back(1);
            realIndices.emplace_back(i);
        } else {
            masks.emplace_back(0);
        }
    }

    AscendTensor inputAt(input), outAt(*out);
    if (realIndices.empty()) {
        return diopiSuccess;
    }

    /**
     * When input.size(0) = 1, if the dtype of indices is int64,
     * and indices only for 0 dimension, can broadcast to output.
     */
    if (inputAt.shape(0) == 1 && masks.size() == 1 && masks[0] == 1 && (!realIndices.empty()) && realIndices[0].dim() == 1) {
        std::vector<int64_t> outputSize = inputAt.shape();
        outputSize[0] = realIndices[0].shape(0);
        AscendTensor result(*out);
        makeTensor(ctx, result, outputSize, inputAt.dtype());
        broadcast(ctx, result, inputAt, outputSize);
        return diopiSuccess;
    }

    std::vector<int64_t> outputSize = indexOutputSize(ctx, inputAt, indicesVec);
    makeTensor(ctx, outAt, outputSize, inputAt.dtype());
    AclOpRunner<4, 1>("Index", ctx).addInput(input).addConstInput(masks).addConstInput(outputSize).addDynamicInput(realIndices).addOutput(outAt).run();

    auto ptr = const_cast<diopiTensorHandle_t>(outAt.tensorHandle());
    *out = ptr;
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
