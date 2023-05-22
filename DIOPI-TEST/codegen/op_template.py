from code_template import CodeTemplate

class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#include <pybind11/pybind11.h>
#include "litert.hpp"
#ifdef TEST_USE_ADAPTOR
#include <diopi_adaptors.hpp>
#endif
#include <diopi/diopirt.h>
namespace py = pybind11;

PYBIND11_MODULE(diopi_functions, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring
    m.def("diopiGetVendorName", &diopiGetVendorName);
    m.def("diopiGetImplVersion", &diopiGetImplVersion);
    m.def("diopiGetVersion", &diopiGetVersion);
    m.def("diopiGetLastErrorString", &diopiGetLastErrorString);
    ${export_functions}
}

""")

    function_template = CodeTemplate("""\
m.def("${func_name}", [](${attrs}) {
    ${convert}
    diopiError_t ret = ${call_func};
    ${out_copy}
    return ret;
});
""")
    
    vector_template = CodeTemplate("""\
std::vector<${handle_type}> ${param}V(${param_num});
for (int i = 0; i < ${param_num}; ++i)
    ${param}V[i] = ${param}[i].cast<PtrWrapper<diopiTensor>>().get();
auto ${param}DIOPI = ${param}V.data();
""")