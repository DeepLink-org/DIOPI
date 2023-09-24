# Copyright (c) 2023, DeepLink.
from codegen.code_template import CodeTemplate


class CaseTemplate:
    # class
    test_class_templ = CodeTemplate(r'''
import os
import pickle
import numpy as np
from conformance.diopi_runtime import Tensor
from conformance.diopi_functions import ones_like
from conformance.check_result import CheckResult
${test_diopi_head_import}

data_path = './cache/data'

class ${test_class_name}(object):
    ${test_case_items}
''')

    # import
    test_diopi_function_import = CodeTemplate(r'''
from conformance.diopi_functions import ${test_diopi_func_name}
${test_import_diopi_bp_func}
''')

    # import
    test_diopi_manual_import = CodeTemplate(r'''
from conformance.diopi_manual_functions import ManualTest
''')

    # test_case
    test_function_templ = CodeTemplate(r'''
def test_${func_case_name}(self):
    ${forward}
    ${backward}
    ${forward_inp}
''')

    test_function_body_forward = CodeTemplate(r'''
f_in = os.path.join(data_path, '${test_module_name}', 'inputs', '${input_data_path}')
${test_function_ref_data_path}

# read input from file
with open(f_in, 'rb') as f:
    data_in = pickle.load(f)

function_paras = data_in['function_paras']
function_kwargs = function_paras['kwargs']

for para_key, para_val in function_kwargs.items():
    if isinstance(para_val, np.ndarray):
        function_kwargs[para_key] = Tensor.from_numpy(para_val)
# output of device: dev_out
${test_function_forward_call}
''')
    test_diopi_function_forward_call = CodeTemplate(r'''
tol = ${test_caompare_tol}
function_config = data_in['cfg']
dev_out = ${test_diopi_func_name}(**function_kwargs)

# read ref_out 
with open(f_out, 'rb') as f:
    ref_out = pickle.load(f)

try:
    CheckResult.compare_output(dev_out, ref_out, **tol)
except Exception as e:
    print(f'Test {function_config["name"]}: {function_config}')
    print(f'{e}')
    assert False
''')

    test_diopi_function_inp_forward_call = CodeTemplate(r'''
# inplace call for the function
${test_diopi_func_inp_remove_grad_args}
function_kwargs.update({'inplace': True})
dev_inp_out = ${test_diopi_func_name}(**function_kwargs)
try:
    CheckResult.compare_output(dev_inp_out, ref_out, **tol)
except Exception as e:
    print(f'Test {function_config["name"]}  inplace: {function_config}')
    print(f'{e}')
    assert False
''')
    test_diopi_func_inp_remove_grad_args = CodeTemplate(r'''
function_kwargs.pop(*backward_para.keys())
''')

    test_manual_function_forward_call = CodeTemplate(r'''
ManualTest.test_${test_diopi_func_name}(**function_kwargs)
''')
    test_manual_function_inp_forward_call = CodeTemplate(r'''
${test_diopi_func_inp_remove_grad_args}
function_kwargs.update({'inplace': True})
ManualTest.test_${test_diopi_func_name}(**function_kwargs)
''')

    #backward
    test_function_body_backward = CodeTemplate(r'''
# grad_output_path
f_bp_out = os.path.join(data_path, '${test_module_name}', 'outputs', '${bp_output_data_path}')

if not isinstance(dev_out, (list, tuple)):
    dev_out = [dev_out]
requires_backward = function_config['requires_backward']
outputs_for_backward = dev_out if len(requires_backward) == 0 \
    else [dev_out[i] for i in requires_backward]
backward_para = {}
grad_outputs = [ones_like(i) for i in outputs_for_backward]
backward_para["grad_outputs"] = grad_outputs
function_kwargs.update(backward_para)
dev_bp_out = ${test_diopi_bp_func_name}(**function_kwargs)
                                           
with open(f_bp_out, 'rb') as f:
    ref_bp_out = pickle.load(f)

# checkout
try:
    CheckResult.compare_output(dev_bp_out, ref_bp_out, **tol)
except Exception as e:
    print(f'Test {function_config["name"]} backward: {function_config}')
    print(e)
    assert False
''')
