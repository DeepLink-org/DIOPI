# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate


class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

//NOLINTBEGIN
#include <pybind11/pybind11.h>
#include "litert.hpp"
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>

namespace py = pybind11;

PYBIND11_MODULE(export_functions, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring
    m.def("diopiGetVendorName", &diopiGetVendorName);
    m.def("diopiGetImplVersion", &diopiGetImplVersion);
    m.def("diopiGetVersion", &diopiGetVersion);
    m.def("diopiGetLastErrorString", &diopiGetLastErrorString);
    ${export_functions}
}
// NOLINTEND
""")

    function_template = CodeTemplate("""\
m.def("${func_name}", [](${attrs}) {
    if (${func_name}) {
        ${convert}
        diopiError_t ret = ${call_func};
        ${out_copy}
        return ret;
    } else {
        return diopiError_t::diopiNoImplement;
    }
});
""")

    vector_template = CodeTemplate("""\
std::vector<${handle_type}> ${param}V(${param_num});
for (int i = 0; i < ${param_num}; ++i)
    ${param}V[i] = ${param}[i].cast<PtrWrapper<diopiTensor>>().get();
auto ${param}DIOPI = ${param}V.data();
""")

    var_len_array_out_template = CodeTemplate("""\
for (int i = 0; i < ${param_num}; ++i) {
    ${param}[i] = ${param}DIOPI[i];
}
""")
