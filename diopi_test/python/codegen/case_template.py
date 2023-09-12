# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate


class CaseTemplate:
    # class
    test_class_templ = CodeTemplate(r'''
import os
import pickle
import numpy as np
${test_diopi_head_import}

data_path = './cache/data'

class ${test_class_name}(object):
    ${test_case_items}
''')

    # import
    test_diopi_head_import = CodeTemplate(r'''
from conformance.diopi_runtime import Tensor
from conformance.diopi_functions import ones_like
from conformance.${test_diopi_function_module} import ${test_diopi_func_name}
# from conformance.diopi_functions import {test_diopi_bp_func_name}
${test_import_diopi_bp_func}
from conformance.check_result import CheckResult
''')

    # test_case
    test_function_templ = CodeTemplate(r'''
def test_${func_case_name}(self):
    ${forward}
    ${backward}
''')

    test_function_body_forward = CodeTemplate(r'''
f_in = os.path.join(data_path, "inputs", '${input_data_path}')
f_out = os.path.join(data_path, "outputs", '${output_data_path}')

# read input from file
with open(f_in, 'rb') as f:
    data_in = pickle.load(f)

function_paras = data_in['function_paras']
function_kwargs = function_paras['kwargs']
function_config = data_in['cfg']

tol = ${test_caompare_tol}

for para_key, para_val in function_kwargs.items():
    if isinstance(para_val, np.ndarray):
        function_kwargs[para_key] = Tensor.from_numpy(para_val)
# output of device 
dev_out = ${test_diopi_func_name}(**function_kwargs)
${test_function_forward_ref_compare}
''')
    test_function_forward_ref_compare = CodeTemplate('''

with open(f_out, 'rb') as f:
    ref_out = pickle.load(f)
CheckResult.compare_output(dev_out, ref_out, **tol)
''')


    #backward
    test_function_body_backward = CodeTemplate(r'''
# grad_output_path
f_bp_out = os.path.join(data_path, 'outputs', '${bp_output_data_path}')

if not isinstance(dev_out, (list, tuple)):
    dev_out = [dev_out]
requires_backward = function_config['requires_backward']
outputs_for_backward = dev_out if len(requires_backward) == 0 \
    else [dev_out[i] for i in requires_backward]
backward_para = {}
grad_outputs = [ones_like(i) for i in outputs_for_backward]
backward_para["grad_outputs"] = grad_outputs

dev_bp_out = ${test_diopi_bp_func_name}(**function_kwargs, **grad_outputs)
                                           
with open(f_bp_out, 'rb') as f:
    ref_bp_out = pickle.load(f)

# checkout
CheckResult.compare_output(dev_bp_out, ref_bp_out, **tol)
''')
