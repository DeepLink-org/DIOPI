# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate

class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#include "convert.hpp"
#include "impl_functions.hpp"

// NOLINTBEGIN
${cast_strategy}

${adaptors}

// NOLINTEND

""")

    adaptor_template = CodeTemplate("""\
extern "C" diopiError_t diopi${op_name}(${attrs}) {
    ${new_input}
    {
        ${cast_input}
    }

    ${cast_output}
    diopiError_t ret;
    {
        ret = ::impl::${device}::${call_func};
    }
    return ret;
}

""")

    adaptor_with_plugin_template = CodeTemplate("""\
extern "C" diopiError_t diopi${op_name}(${attrs}) {
    ${new_input}
    {
        ${cast_input}
    }

    ${cast_output}
    diopiError_t ret;
    {
        if (::impl::${device}::diopi${op_name}) {
            ret = ::impl::${device}::${call_func};
        } else if (::impl::${device}_npu::diopi${op_name}) {
            ret = ::impl::${device}_npu::${call_func};
        } else {
            ret = diopiNoImplement;
        }
    }
    return ret;
}
""")

    cast_strategy_template = CodeTemplate("""\
class ${cast_name} {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            ${cases}
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};
""")
    impl_declaration_head_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_FUNCTIONS_HPP_
#define IMPL_FUNCTIONS_HPP_

#include <diopi/diopirt.h>

// NOLINTBEGIN
namespace impl {
${impl_declaration_content}
} // namespace impl
// NOLINTEND
#endif  // IMPL_FUNCTIONS_HPP_
""")

    impl_declaration_content_template = CodeTemplate("""\
namespace ${device} {

${impl_declaration}

}  // namespace ${device}
""")
