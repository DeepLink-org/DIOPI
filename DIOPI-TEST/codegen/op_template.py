# Copyright (c) 2023, DeepLink.
import re


class CodeTemplate(object):
    substitution_str = r'(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})'

    # older versions of Python have a bug where \w* does not work,
    # so we need to replace with the non-shortened version [a-zA-Z0-9_]*
    # https://bugs.python.org/issue18647

    substitution_str = substitution_str.replace(r'\w', r'[a-zA-Z0-9_]')

    subtitution = re.compile(substitution_str, re.MULTILINE)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            return CodeTemplate(f.read())

    def __init__(self, pattern):
        self.pattern = pattern

    def substitute(self, env={}, **kwargs):
        def lookup(v):
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent, v):
            return ("".join([indent + line + "\n" for e in v
                    for line in str(e).splitlines()]).rstrip())

        def replace(match):
            indent = match.group(1)
            key = match.group(2)
            comma_before = ''
            comma_after = ''
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ', '
                    key = key[1:]
                if key[-1] == ',':
                    comma_after = ', '
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ', '.join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)
        return self.subtitution.sub(replace, self.pattern)


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
#ifdef TEST_USE_ADAPTOR
#include <diopi/diopi_adaptors.hpp>
#endif
#include <diopi/diopirt.h>
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
        py::gil_scoped_release no_gil;
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
