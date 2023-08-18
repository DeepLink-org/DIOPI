/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
// infer the size of a expanded tensor based on the sizes of two input tensors a and b
static std::vector<int64_t> inferSize(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
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
        assert((sizeA == sizeB || sizeA == 1 || sizeB == 1) && "The size of tensor a must match the size of tensor b at a non-singleton dimension");
        expandedSize[i] = sizeA == 1 ? sizeB : sizeA;
    }
    return expandedSize;
}

// true if all the non-null tensors are adjacent
static bool hasContiguousSubspace(const std::vector<DiopiTensor>& tensorList) {
    auto isDefined = [](const DiopiTensor& tensor) { return tensor.defined(); };
    auto isNull = [](const DiopiTensor& tensor) { return !tensor.defined(); };
    auto start = std::find_if(tensorList.begin(), tensorList.end(), isDefined);
    auto stop = std::find_if(tensorList.rbegin(), tensorList.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

// return a new output tensor with singleton dimensions expanded to a larger size based on the input tensor
static diopiError_t expand(cnnlHandle_t handle, DiopiTensor outTensor, DiopiTensor inputTensor) {
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

// compared to the input tensor, the dimensions of the output tensor are permuted according to the given order
static diopiError_t permute(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, std::vector<int32_t> order) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> transDesc;
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(transDesc.get(), inputTensor.dim(), order.data()));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, inputDesc.get(), transDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTranspose_v2(handle, transDesc.get(), inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data(), workspace, workspaceSize));
    return diopiSuccess;
}

static diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor* numTrue) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<int64_t> shape{1};
    *numTrue = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc numTrueDesc(*numTrue, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlNumTrue_v2(handle, inputDesc.get(), inputTensor.data(), numTrueDesc.get(), numTrue->data()));
    return diopiSuccess;
}

// return a tensor containing the indices of all non-zero elements of input
// assist in converting bool indices to equivalent integer indices, because the indices list of the index op does not support mixed use of different dtypes of
// index tensors
static diopiError_t nonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor numTrue;
    nonzeroCount(ctx, inputTensor, &numTrue);
    CnnlTensorDesc numTrueDesc(numTrue, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetWhereWorkspaceSize(handle, numTrueDesc.get(), &workspaceSize));
    void* workspace = nullptr;
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

// to satisfy the kernel requirement, the bool index tensor in the input indices list is first converted to an equivalent integer index tensor
static diopiError_t indexPreProcess(diopiContextHandle_t ctx, DiopiTensor inputTensor, std::vector<DiopiTensor> indicesTensors,
                                    DiopiTensor& transposedInputTensor, std::vector<DiopiTensor>& transposedIndicesTensors) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // first expand bool tensor (masks) or byte tensor (masks) into 1 or more long tensors
    bool boolTensorConvertToEmptyTensor = false;
    std::vector<DiopiTensor> indicesTensorsCast;
    for (const auto& indiceTensor : indicesTensors) {
        if (!indiceTensor.defined()) {
            indicesTensorsCast.emplace_back();
        } else {
            DiopiTensor indexTensor = indiceTensor;
            if (indexTensor.dtype() == diopi_dtype_uint8 || indexTensor.dtype() == diopi_dtype_bool) {
                // the sizes of byte tensor or bool tensor must match the sizes of the corresponding dimensions in input
                for (auto j = 0; j < indexTensor.dim(); ++j) {
                    int64_t srcIdx = indicesTensorsCast.size() + j;
                    if (indexTensor.size(j) != inputTensor.size(srcIdx)) {
                        DIOPI_CHECK(false, "invalid mask");
                    }
                }
                // replace with nonzeros
                diopiTensorHandle_t out = nullptr;
                DIOPI_CALL(nonzero(ctx, &out, indexTensor.tensorHandle()));
                DiopiTensor nonzeroTensor(out);
                // empty tensor judgment
                if (nonzeroTensor.numel()) {
                    for (auto j = 0; j < indexTensor.dim(); ++j) {
                        std::vector<int64_t> selectShape = nonzeroTensor.shape();
                        selectShape.erase(selectShape.begin() + 1);
                        DiopiTensor selectTensor = requiresTensor(ctx, selectShape, nonzeroTensor.dtype());
                        DIOPI_CALL(diopiSelect(ctx, selectTensor.tensorHandle(), nonzeroTensor.tensorHandle(), 1, j));
                        indicesTensorsCast.emplace_back(std::move(selectTensor));
                    }
                } else {
                    // specical case: bool tensor -> empty int tensor
                    for (auto j = 0; j < indexTensor.dim(); ++j) {
                        std::vector<int64_t> emptyShape{0};
                        DiopiTensor emptyTensor = requiresTensor(ctx, emptyShape, nonzeroTensor.dtype());
                        indicesTensorsCast.emplace_back(std::move(emptyTensor));
                    }
                    boolTensorConvertToEmptyTensor = true;
                }
            } else {
                // int tensor
                indicesTensorsCast.emplace_back(std::move(indexTensor));
            }
        }
    }
    indicesTensorsCast.resize(indicesTensors.size());

    // next broadcast all index tensors together, ignore undefined (null) tensors
    bool first = true;
    std::vector<int64_t> sizes;
    std::vector<DiopiTensor> indicesTensorsExpand(indicesTensorsCast.size());
    for (auto& indiceTensorCast : indicesTensorsCast) {
        if (!(indiceTensorCast.defined() && indiceTensorCast.numel())) {
            continue;
        } else if (first) {
            sizes = indiceTensorCast.shape();
            first = false;
        } else {
            sizes = inferSize(sizes, indiceTensorCast.shape());
        }
    }
    for (auto i = 0; i < indicesTensorsCast.size(); ++i) {
        if (!(indicesTensorsCast[i].defined() && indicesTensorsCast[i].numel())) {
            if (indicesTensorsCast[i].defined()) {
                // handle the broadcast of empty indice tensor
                indicesTensorsExpand[i] = indicesTensorsCast[i];
                if (boolTensorConvertToEmptyTensor) {
                    std::vector<int64_t> tmpShape(sizes.size(), 1);
                    tmpShape.insert(tmpShape.begin(), 0);
                    indicesTensorsExpand[i].view(tmpShape);
                }
            } else {
                continue;
            }
        } else if (indicesTensorsCast[i].shape() == sizes) {
            indicesTensorsExpand[i] = indicesTensorsCast[i];
        } else {
            DiopiTensor expanded = requiresTensor(ctx, sizes, indicesTensorsCast[i].dtype());
            DIOPI_CALL(expand(handle, expanded, indicesTensorsCast[i]));
            indicesTensorsExpand[i] = expanded;
        }
    }

    // add missing null tensors so that it matches input.dim()
    while (indicesTensorsExpand.size() < inputTensor.dim()) {
        indicesTensorsExpand.emplace_back();
    }

    // transpose input and indices together so that they're adjacent at the front
    if (!hasContiguousSubspace(indicesTensorsExpand)) {
        std::vector<int32_t> order;
        order.reserve(inputTensor.dim());
        std::vector<int64_t> transposedShape = inputTensor.shape();
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            if (indicesTensorsExpand[i].defined()) {
                order.emplace_back(i);
                transposedIndicesTensors.emplace_back(indicesTensorsExpand[i]);
            }
        }
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            if (!indicesTensorsExpand[i].defined()) {
                order.emplace_back(i);
                transposedIndicesTensors.emplace_back();
            }
        }
        for (auto i = 0; i < inputTensor.dim(); ++i) {
            transposedShape[i] = inputTensor.size(order[i]);
        }
        transposedInputTensor = requiresTensor(ctx, transposedShape, inputTensor.dtype());
        // meet cnnl kernel requirements
        while (transposedIndicesTensors.size() < indicesTensors.size()) {
            transposedIndicesTensors.emplace_back();
        }
        DIOPI_CALL(permute(ctx, transposedInputTensor, inputTensor, order));
    } else {
        transposedInputTensor = inputTensor;
        transposedIndicesTensors = indicesTensorsExpand;
    }
    return diopiSuccess;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
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
            if (indiceTensor.defined()) {
                if (indiceTensor.dtype() == diopi_dtype_int64) {
                    DIOPI_CALL(dataTypeCast(ctx, indiceTensor, diopi_dtype_int32));
                }
                DIOPI_CHECK(indiceTensor.dtype() == diopi_dtype_int32 || indiceTensor.dtype() == diopi_dtype_bool || indiceTensor.dtype() == diopi_dtype_uint8,
                            "Tensors used as indices must be int32, bool or uint8 tensors");
                if (indiceTensor.numel()) {
                    indicesTensors.emplace_back(indiceTensor);
                } else {
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

    DiopiTensor transposedInputTensor;
    std::vector<DiopiTensor> transposedIndicesTensors;
    DIOPI_CALL(indexPreProcess(ctx, inputTensorTmp, indicesTensors, transposedInputTensor, transposedIndicesTensors));
    CnnlTensorDesc inputDesc(transposedInputTensor, CNNL_LAYOUT_ARRAY);

    std::vector<CnnlTensorDesc> indicesDesc(arraySize);
    std::vector<cnnlTensorDescriptor_t> indicesDescT(arraySize);
    std::vector<void*> indicesPtrList(arraySize);
    for (auto i = 0; i < arraySize; ++i) {
        if (transposedIndicesTensors[i].defined()) {
            indicesDesc[i].set(transposedIndicesTensors[i], CNNL_LAYOUT_ARRAY);
            indicesDescT[i] = indicesDesc[i].get();
            indicesPtrList[i] = transposedIndicesTensors[i].data();
        } else {
            indicesDescT[i] = nullptr;
            indicesPtrList[i] = nullptr;
        }
    }

    int32_t outputDescDim = 0;
    std::vector<int32_t> outputDescDims(arraySize);
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexOutputDim(handle, inputDesc.get(), indicesDescT.data(), &outputDescDim, outputDescDims.data()));
    outputDescDims.resize(outputDescDim);

    std::vector<int64_t> outTensorShape(outputDescDims.begin(), outputDescDims.end());
    DiopiTensor outTensor = requiresTensor(ctx, outTensorShape, inputTensorTmp.dtype());
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DiopiTensor outputDims = requiresTensor(ctx, {static_cast<int64_t>(outTensorShape.size())}, diopi_dtype_int64);
    cnrtMemcpy(outputDims.data(), outTensorShape.data(), sizeof(int64_t) * outTensorShape.size(), cnrtMemcpyHostToDev);
    DiopiTensor outputDim = requiresTensor(ctx, {1}, diopi_dtype_int32);
    cnrtMemcpy(outputDim.data(), &outputDescDim, sizeof(int32_t), cnrtMemcpyHostToDev);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAdvancedIndexWorkspaceSize(handle, inputDesc.get(), indicesDescT.data(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlAdvancedIndex(handle,
                                     inputDesc.get(),
                                     transposedInputTensor.data(),
                                     indicesDescT.data(),
                                     indicesPtrList.data(),
                                     workspace,
                                     workspaceSize,
                                     outDesc.get(),
                                     outTensor.data(),
                                     outputDims.data(),
                                     outputDim.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, inputTensor.dtype()));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
