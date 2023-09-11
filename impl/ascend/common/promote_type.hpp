/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_PROMOTE_TYPE_HPP_
#define IMPL_ASCEND_COMMON_PROMOTE_TYPE_HPP_

#include <diopi/diopirt.h>

#include <map>

#include "../error.hpp"

namespace impl {
namespace ascend {

static inline diopiDtype_t promoteTypes(diopiDtype_t a, diopiDtype_t b) {
    // This is generated according to NumPy's promote_types
    constexpr auto u1 = diopi_dtype_uint8;
    constexpr auto i1 = diopi_dtype_int8;
    constexpr auto i2 = diopi_dtype_int16;
    constexpr auto i4 = diopi_dtype_int32;
    constexpr auto i8 = diopi_dtype_int64;
    constexpr auto f2 = diopi_dtype_float16;
    constexpr auto f4 = diopi_dtype_float32;
    constexpr auto f8 = diopi_dtype_float64;
    constexpr auto c4 = diopi_dtype_complex64;
    constexpr auto c8 = diopi_dtype_complex128;
    constexpr auto b1 = diopi_dtype_bool;

    static std::map<diopiDtype_t, int> dtypeMap = {{u1, 0}, {i1, 1}, {i2, 2}, {i4, 3}, {i8, 4}, {f2, 5}, {f4, 6}, {f8, 7}, {c4, 8}, {c8, 9}, {b1, 10}};
    static constexpr diopiDtype_t promoteTypesLookup[11][11] = {
        /*        u1  i1  i2  i4  i8  f2  f4  f8  c4  c8  b1*/
        /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c4, c8, u1},
        /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c4, c8, i1},
        /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c4, c8, i2},
        /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c4, c8, i4},
        /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c4, c8, i8},
        /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c4, c8, f2},
        /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, f4},
        /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, f8},
        /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c8, c4},
        /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
        /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c4, c8, b1},
    };

    ASCEND_CHECK_ABORT((dtypeMap.count(a) != 0 && dtypeMap.count(b) != 0), "dtype a %d or b %d not supported.", a, b);
    return promoteTypesLookup[dtypeMap[a]][dtypeMap[b]];
}

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_COMMON_PROMOTE_TYPE_HPP_
