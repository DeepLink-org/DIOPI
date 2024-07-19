/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <cstdint>
#include "helper.hpp"
#include "op_plugin/OpApiInterface.h"

namespace OP_IMPL_NS {

diopiError_t diopiUnique1(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    bool return_inverse = (indices != nullptr) ? true : false;
    BEGIN_CALL_ACL_OP(input);
    c10::optional<int64_t> dimAt = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    at::Tensor y, y_inverse, y_counts;
    std::tie(y, y_inverse, y_counts) = op_api::unique_consecutive(inputAt, return_inverse, return_counts, dimAt);
    
    impl::aten::buildDiopiTensor(ctx, y, out);
    if (return_inverse) {
        impl::aten::buildDiopiTensor(ctx, y_inverse, &indices);
    }
    if (return_counts) {
        impl::aten::buildDiopiTensor(ctx, y_counts, counts);
    }
    END_CALL_ACL_OP();
}



diopiError_t diopiUnique2(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    bool return_inverse = (indices != nullptr) ? true : false;
    BEGIN_CALL_ACL_OP(input);
    c10::optional<int64_t> dimAt = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    at::Tensor y, y_inverse, y_counts;
    std::tie(y, y_inverse, y_counts) = op_api::_unique2(inputAt, sorted, return_inverse, return_counts);
    
    impl::aten::buildDiopiTensor(ctx, y, out);
    if (return_inverse) {
        impl::aten::buildDiopiTensor(ctx, y_inverse, &indices);
    }
    if (return_counts) {
        impl::aten::buildDiopiTensor(ctx, y_counts, counts);
    }
    END_CALL_ACL_OP();
}


diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {

    if (dim) {
        return diopiUnique1(ctx, out, input, dim, sorted, return_counts, indices, counts);
    } else {
        return diopiUnique2(ctx, out, input, dim, sorted, return_counts, indices, counts);
    }
    return diopiSuccess;
}


}  // namespace OP_IMPL_NS



