# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate

class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_ADAPTOR_HPP_
#define DIOPI_ADAPTOR_HPP_

#include "convert.hpp"
#include "impl_functions.hpp"

${cast_strategy}

${adaptors}

# endif // DIOPI_ADAPTOR_HPP
""")

    adaptor_template = CodeTemplate("""\
extern "C" diopiError_t diopi${op_name}(${attrs}) {
    TimeElapsed adaptorTimeElapsed("${op_name}_adaptor");
    ${new_input}
    {
        TimeElapsed castInputTimeElapsed("${op_name}_cast_input");
        ${cast_input}
    }

    ${cast_output}
    diopiError_t ret;
    {
        TimeElapsed opTimeElapsed("${op_name}");
        if(::impl::${device}::${func_name}) {
            ret = ::impl::${device}::${call_func};
        }
        else {
            return diopiError_t::diopiNoImplement;
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

    impl_declaration_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_FUNCTIONS_HPP_
#define IMPL_FUNCTIONS_HPP_

#include <diopi/diopirt.h>

namespace impl {
namespace ${device} {

${impl_declaration}

}  // namespace ${device}
}  // namespace impl

#endif  // IMPL_FUNCTIONS_HPP_

""")
