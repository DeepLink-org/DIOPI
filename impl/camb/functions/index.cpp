/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {

static std::vector<int64_t> inferSize(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
    int32_t dimsA = a.size();
    int32_t dimsB = b.size();
    int32_t ndim = dimsA > dimsB ? dimsA : dimsB;
    std::vector<int64_t> expandedSize(ndim);
    for (auto i = ndim - 1; i >= 0; --i) {
        auto offset = ndim - 1 - i;
        auto dimA = dimsA - 1 - offset;
        auto dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? a[dimA] : 1;
        auto sizeB = (dimB >= 0) ? b[dimB] : 1;
        assert(sizeA == sizeB || sizeA == 1 || sizeB == 1 && "The size of tensor a must match the size of tensor b at a non-singleton dimension");
        expandedSize[i] = sizeA == 1 ? sizeB : sizeA;
    }
    return expandedSize;
}

// true if all the non-null tensors are adjacent
static bool hasContiguousSubspace(const std::vector<DiopiTensor> &tensorList) {
    auto isDefined = [](const DiopiTensor &tensor) { return static_cast<bool>(tensor.tensorHandle()); };
    auto isNull = [](const DiopiTensor &tensor) { return static_cast<bool>(!tensor.tensorHandle()); };
    auto start = std::find_if(tensorList.begin(), tensorList.end(), isDefined);
    auto stop = std::find_if(tensorList.rbegin(), tensorList.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

static diopiError_t expand(cnnlHandle_t handle, DiopiTensor outTensor, DiopiTensor inputTensor) {
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

static diopiError_t permute(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, std::vector<int32_t> order) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> transDesc;
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transDesc.get(), inputTensor.dim(), order.data()));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

static diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor *numTrue) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<int64_t> shape = {1};
    *numTrue = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc numTrueDesc(*numTrue, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlNumTrue_v2(handle, inputDesc.get(), inputTensor.data(), numTrueDesc.get(), numTrue->data()));
    return diopiSuccess;
}

static diopiError_t nonzero(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor numTrue;
    nonzeroCount(ctx, inputTensor, &numTrue);
    CnnlTensorDesc numTrueDesc(numTrue, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetWhereWorkspaceSize(handle, numTrueDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    syncStreamInCtx(ctx);
    int32_t count = 0;
    cnrtMemcpy(&count, numTrue.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);

    std::vector<int64_t> shape(2);
    shape[0] = count;
    shape[1] = inputTensor.dim();
    DiopiTensor outTensor = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlWhere_v2(
        handle, inputDesc.get(), inputTensor.data(), numTrueDesc.get(), numTrue.data(), false, workspace, workspaceSize, outDesc.get(), outTensor.data()));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

static diopiError_t indexPreProcess(diopiContextHandle_t ctx, DiopiTensor inputTensor, std::vector<DiopiTensor> indices, DiopiTensor &transposedInput,
                                    std::vector<DiopiTensor> &transposedIndices, bool &indiceEmptyTensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // first expand bool tensor (masks) or byte tensor (masks) into 1 or more long tensors
    std::vector<DiopiTensor> indicesCast;
    for (auto indice : indices) {
        if (!indice.tensorHandle()) {
            indicesCast.emplace_back();
        } else {
            DiopiTensor index = std::move(indice);
            if (index.dtype() == diopi_dtype_uint8 || index.dtype() == diopi_dtype_bool) {
                // the sizes of byte tensor or bool tensor must match the sizes of the corresponding dimensions in input
                for (auto j = 0; j < index.dim(); ++j) {
                    int64_t srcIdx = indicesCast.size() + j;
                    if (index.size(j) != inputTensor.size(srcIdx)) {
                        DIOPI_CHECK(false, "invalid mask");
                    }
                }
                // replace with nonzeros
                diopiTensorHandle_t out = nullptr;
                DIOPI_CALL(nonzero(ctx, &out, index.tensorHandle()));
                DiopiTensor nonzeroTensor(out);
                // empty tensor judgment
                if (nonzeroTensor.numel() != 0) {
                    for (auto j = 0; j < index.dim(); j++) {
                        std::vector<int64_t> selectShape = nonzeroTensor.shape();
                        selectShape.erase(selectShape.begin() + 1);
                        DiopiTensor selectTensor = requiresTensor(ctx, selectShape, nonzeroTensor.dtype());
                        DIOPI_CALL(diopiSelect(ctx, selectTensor.tensorHandle(), nonzeroTensor.tensorHandle(), 1, j));
                        indicesCast.emplace_back(std::move(selectTensor));
                    }
                } else {
                    // specical case: bool tensor -> empty tensor
                    for (auto j = 0; j < index.dim(); j++) {
                        DiopiTensor emptyTensor = requiresTensor(ctx, {0}, nonzeroTensor.dtype());
                        indicesCast.emplace_back(std::move(emptyTensor));
                    }
                    indiceEmptyTensor = true;
                }
            } else {
                // int tensor
                if (index.numel()) {
                    indicesCast.emplace_back(std::move(index));
                } else {
                    indiceEmptyTensor = true;
                    indicesCast.emplace_back(std::move(index));
                }
            }
        }
    }
    indicesCast.resize(indices.size());

    // next broadcast all index tensors together, ignore undefined (null) tensors
    bool first = true;
    std::vector<int64_t> sizes;
    std::vector<DiopiTensor> indicesExpand(indicesCast.size());
    for (auto i = 0; i < indicesCast.size(); ++i) {
        if (!indicesCast[i].defined()) {
            continue;
        } else if (first) {
            sizes = indicesCast[i].shape();
            first = false;
        } else {
            sizes = inferSize(sizes, indicesCast[i].shape());
        }
    }
    for (auto i = 0; i < indicesCast.size(); ++i) {
        if (!indicesCast[i].defined()) {
            if (indicesCast[i].tensorHandle()) {
                // handle the broadcast of empty indice tensor
                // std::cout << "empty tensor处理" << std::endl;
                indicesExpand[i] = indicesCast[i];
                std::vector<int64_t> tmpShape(sizes.size(), 1);
                tmpShape.insert(tmpShape.begin(), 0);
                indicesExpand[i].reshape(tmpShape);
            } else {
                continue;
            }
        } else if (indicesCast[i].shape() == sizes) {
            indicesExpand[i] = indicesCast[i];
        } else {
            DiopiTensor expanded = requiresTensor(ctx, sizes, indicesCast[i].dtype());
            DIOPI_CALL(expand(handle, expanded, indicesCast[i]));
            indicesExpand[i] = expanded;
        }
    }

    // add missing null tensors so that it matches input.dim()
    while (indicesExpand.size() < inputTensor.dim()) {
        indicesExpand.emplace_back();
    }

    // transpose input and indices together so that they're adjacent at the front
    if (!hasContiguousSubspace(indicesExpand)) {
        std::vector<int32_t> order;
        order.reserve(inputTensor.dim());
        std::vector<int64_t> transposedShape = inputTensor.shape();
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            if (indicesExpand[i].tensorHandle()) {
                order.emplace_back(i);
                transposedIndices.emplace_back(indicesExpand[i]);
            }
        }
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            if (!indicesExpand[i].tensorHandle()) {
                order.emplace_back(i);
                transposedIndices.emplace_back();
            }
        }
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            transposedShape[i] = inputTensor.size(order[i]);
        }
        transposedInput = requiresTensor(ctx, transposedShape, inputTensor.dtype());
        // meet cnnl kernel requirements
        while (transposedIndices.size() < indices.size()) {
            transposedIndices.emplace_back();
        }
        DIOPI_CALL(permute(ctx, transposedInput, inputTensor, order));
    } else {
        transposedInput = inputTensor;
        transposedIndices = indicesExpand;
    }
    return diopiSuccess;
}

static diopiError_t computeOutputShape(diopiContextHandle_t ctx, const DiopiTensor &transposedInput, const std::vector<DiopiTensor> &transposedIndices,
                                       std::vector<int64_t> &cnnlOutTensorShape, std::vector<int64_t> &actualOutTensorShape) {
    int64_t dimsBefore = 0, dimsAfter = 0, dimsIndexed = 0;
    std::vector<int64_t> replacementShape;
    std::vector<int64_t> actualReplacementShape;
    std::vector<int64_t> indexedSizes;
    for (auto dim = 0; dim < transposedIndices.size(); ++dim) {
        if (!transposedIndices[dim].tensorHandle()) {
            if (dimsIndexed == 0) {
                dimsBefore++;
            } else {
                dimsAfter++;
            }
        } else {
            dimsIndexed++;
            replacementShape = transposedIndices[dim].shape();
            // handle empty indice tensor
            if (transposedIndices[dim].numel()) {
                actualReplacementShape = transposedIndices[dim].shape();
            } else {
                actualReplacementShape = {0};
            }
            indexedSizes.emplace_back(transposedInput.shape()[dim]);
        }
    }
    // check if the indexed subspace contains a dim of size 0, but the replacement shape does not.
    if (std::find(indexedSizes.begin(), indexedSizes.end(), 0) != indexedSizes.end() &&
        std::find(replacementShape.begin(), replacementShape.end(), 0) == replacementShape.end()) {
        DIOPI_CHECK(false, "index is out of bounds for dimension with size 0");
    }
    if (std::find(indexedSizes.begin(), indexedSizes.end(), 0) != indexedSizes.end() &&
        std::find(actualReplacementShape.begin(), actualReplacementShape.end(), 0) == actualReplacementShape.end()) {
        DIOPI_CHECK(false, "index is out of bounds for dimension with size 0");
    }

    cnnlOutTensorShape = transposedInput.shape();
    actualOutTensorShape = transposedInput.shape();
    int64_t end = dimsBefore + dimsIndexed;
    cnnlOutTensorShape.erase(cnnlOutTensorShape.begin() + dimsBefore, cnnlOutTensorShape.begin() + end);
    cnnlOutTensorShape.insert(cnnlOutTensorShape.begin() + dimsBefore, replacementShape.begin(), replacementShape.end());
    actualOutTensorShape.erase(actualOutTensorShape.begin() + dimsBefore, actualOutTensorShape.begin() + end);
    actualOutTensorShape.insert(actualOutTensorShape.begin() + dimsBefore, actualReplacementShape.begin(), actualReplacementShape.end());
    return diopiSuccess;
}

extern "C" {

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t *indices, int64_t nums) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    const int64_t arraySize = 8;

    DiopiTensor inputTensor(input);
    DiopiTensor inputTensorTmp = inputTensor;
    DIOPI_CHECK(nums <= inputTensor.dim(), "Too many indices for tensor of dimension");
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensorTmp}, supportedDtypes));
    std::vector<DiopiTensor> indicesTensors;
    for (auto i = 0; i < arraySize; ++i) {
        if (i < nums) {
            DiopiTensor indiceTensor(indices[i]);
            if (indiceTensor.tensorHandle()) {
                if (indiceTensor.dtype() == diopi_dtype_int64) {
                    DIOPI_CALL(dataTypeCast(ctx, indiceTensor, diopi_dtype_int32));
                }
                DIOPI_CHECK(indiceTensor.dtype() == diopi_dtype_int32 || indiceTensor.dtype() == diopi_dtype_bool || indiceTensor.dtype() == diopi_dtype_uint8,
                            "Tensors used as indices must be int32, bool or uint8 tensors");
                if (indiceTensor.numel()) {
                    indicesTensors.emplace_back(indiceTensor);
                } else {
                    // DiopiTensor emptyTensor = requiresTensor(ctx, {0}, indiceTensor.dtype());
                    std::cout << "empty indiceTesor.shape: ";
                    for (auto num : indiceTensor.shape()) {
                        std::cout << num << ",";
                    }
                    std::cout << std::endl;
                    DiopiTensor emptyTensor = requiresTensor(ctx, indiceTensor.shape(), indiceTensor.dtype());
                    indicesTensors.emplace_back(std::move(emptyTensor));
                }
            } else {
                indicesTensors.emplace_back();
            }
        } else {
            indicesTensors.emplace_back();
        }
    }

    bool indiceEmptyTensor = false;
    DiopiTensor transposedInput;
    std::vector<DiopiTensor> transposedIndices;
    std::vector<int64_t> cnnlOutTensorShape;
    std::vector<int64_t> actualOutTensorShape;
    DIOPI_CALL(indexPreProcess(ctx, inputTensorTmp, indicesTensors, transposedInput, transposedIndices, indiceEmptyTensor));
    CnnlTensorDesc inputDesc(transposedInput, CNNL_LAYOUT_ARRAY);

    std::vector<CnnlTensorDesc> indicesDesc(arraySize);
    std::vector<cnnlTensorDescriptor_t> indicesDescT(arraySize);
    std::vector<void *> indicesPtrList(arraySize);
    for (auto i = 0; i < arraySize; ++i) {
        if (transposedIndices[i].tensorHandle()) {
            indicesDesc[i].set(transposedIndices[i], CNNL_LAYOUT_ARRAY);
            indicesDescT[i] = indicesDesc[i].get();
            indicesPtrList[i] = transposedIndices[i].data();
        } else {
            indicesDescT[i] = nullptr;
            indicesPtrList[i] = nullptr;
        }
    }

    DIOPI_CALL(computeOutputShape(ctx, transposedInput, transposedIndices, cnnlOutTensorShape, actualOutTensorShape));
    int32_t outputDescDim = cnnlOutTensorShape.size();
    DiopiTensor outTensor = requiresTensor(ctx, cnnlOutTensorShape, inputTensorTmp.dtype());
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor outputDims = requiresTensor(ctx, {static_cast<int64_t>(cnnlOutTensorShape.size())}, diopi_dtype_int64);
    cnrtMemcpy(outputDims.data(), cnnlOutTensorShape.data(), sizeof(int64_t) * cnnlOutTensorShape.size(), cnrtMemcpyHostToDev);
    DiopiTensor outputDim = requiresTensor(ctx, {1}, diopi_dtype_int32);
    cnrtMemcpy(outputDim.data(), &outputDescDim, sizeof(int32_t), cnrtMemcpyHostToDev);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexWorkspaceSize(handle, inputDesc.get(), indicesDescT.data(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlAdvancedIndex(handle,
                                     inputDesc.get(),
                                     transposedInput.data(),
                                     indicesDescT.data(),
                                     indicesPtrList.data(),
                                     workspace,
                                     workspaceSize,
                                     outDesc.get(),
                                     outTensor.data(),
                                     outputDims.data(),
                                     outputDim.data()));
    if (indiceEmptyTensor) {
        std::cout << "indiceEmptyTensor" << std::endl;
        outTensor.reshape(actualOutTensorShape);
    }
    std::cout << "cnnloutshape: ";
    for (auto num : cnnlOutTensorShape) {
        std::cout << num << ",";
    }
    std::cout << std::endl;
    std::cout << "actualoutshape: ";
    for (auto num : actualOutTensorShape) {
        std::cout << num << ",";
    }
    std::cout << std::endl;
    DIOPI_CALL(dataTypeCast(ctx, outTensor, inputTensor.dtype()));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
