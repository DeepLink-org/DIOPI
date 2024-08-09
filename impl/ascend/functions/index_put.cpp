/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

namespace indexProcess {
extern std::vector<AscendTensor> castIntIndicesToLongIndices(diopiContextHandle_t ctx, std::vector<AscendTensor>& indices);
extern void checkIndexTensorTypes(const std::vector<AscendTensor>& indices);
extern std::vector<AscendTensor> expandIndicesTensors(diopiContextHandle_t ctx, const AscendTensor& self, const std::vector<AscendTensor>& indices);
extern aclTensor* createEmptyAclTensor();
}  // namespace indexProcess

diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                           diopiConstTensorHandle_t* indices, int64_t indicesCounts, bool accumulate) {
    diopiCopyInp(ctx, input, out);
    AscendTensor inputAt(input);
    AscendTensor valuesAt(values);
    if (inputAt.numel() == 0 || valuesAt.numel() == 0) {
        return diopiSuccess;
    }
    std::vector<AscendTensor> indicesOrigin(indicesCounts);
    for (int64_t i = 0; i < indicesCounts; i++) {
        if (indices[i] != nullptr) {
            indicesOrigin[i] = AscendTensor(indices[i]);
        }
    }
    std::vector<AscendTensor> indicesList = indexProcess::castIntIndicesToLongIndices(ctx, indicesOrigin);
    indexProcess::checkIndexTensorTypes(indicesList);
    auto indicesExpanded = indexProcess::expandIndicesTensors(ctx, inputAt, indicesList);
    std::vector<aclTensor*> allDefinedIndices;
    auto emptyTensor = indexProcess::createEmptyAclTensor();
    for (const auto& idx : indicesExpanded) {
        if (idx.defined()) {
            allDefinedIndices.push_back(aclnn_adaptor::createAclTensorFromAscendTensor(idx));
        } else {
            allDefinedIndices.push_back(emptyTensor);
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, out, allDefinedIndices, values, accumulate, false);
    return diopiSuccess;
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indicesCounts, bool accumulate) {
    AscendTensor inputAt(input);
    AscendTensor valuesAt(values);
    if (inputAt.numel() == 0 || valuesAt.numel() == 0) {
        return diopiSuccess;
    }
    std::vector<AscendTensor> indicesOrigin(indicesCounts);
    for (int64_t i = 0; i < indicesCounts; i++) {
        if (indices[i] != nullptr) {
            indicesOrigin[i] = AscendTensor(indices[i]);
        }
    }
    std::vector<AscendTensor> indicesList = indexProcess::castIntIndicesToLongIndices(ctx, indicesOrigin);
    indexProcess::checkIndexTensorTypes(indicesList);
    auto indicesExpanded = indexProcess::expandIndicesTensors(ctx, inputAt, indicesList);
    std::vector<aclTensor*> allDefinedIndices;
    auto emptyTensor = indexProcess::createEmptyAclTensor();
    for (const auto& idx : indicesExpanded) {
        if (idx.defined()) {
            allDefinedIndices.push_back(aclnn_adaptor::createAclTensorFromAscendTensor(idx));
        } else {
            allDefinedIndices.push_back(emptyTensor);
        }
    }

    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexPutImpl, ctx, input, allDefinedIndices, values, accumulate, false);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
