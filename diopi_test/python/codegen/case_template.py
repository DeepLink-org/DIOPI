# Copyright (c) 2023, DeepLink.
from codegen.code_template import CodeTemplate


class CaseTemplate:
    # class
    test_class_templ = CodeTemplate(
        r"""
import os
import pickle
import pytest
#import psutil
import numpy as np
from copy import deepcopy

from conformance.diopi_runtime import Tensor, from_numpy_dtype, default_context
from conformance.diopi_functions import ones_like, FunctionNotImplementedError, FunctionNotDefinedError
from conformance.check_result import CheckResult
${test_diopi_head_import}

data_path = './cache'

# @pytest.fixture(scope='class', autouse=True)
# def process_cls():
#     pid = os.getpid()
#     prs = psutil.Process(pid)
#     memory_info = prs.memory_info()
#     print(f'[TestClass] host memory used: {pid} : {memory_info.rss / 1024 / 1024 / 1024} G.')

class ${test_class_name}(object):
    # if run test seprately, this setup and teardown function should be uncommented.
    # from conformance.diopi_runtime import diopi_rt_init
    # @classmethod
    # def setup_dev(cls):
    #     diopi_rt_init()
    #
    # @classmethod
    # def teardown_rt(cls):
    #    pass

    ${test_case_items}
"""
    )

    # import
    test_diopi_function_import = CodeTemplate(
        r"""
from conformance.diopi_functions import ${test_diopi_func_name}
${test_import_diopi_bp_func}
${test_diopi_nhwc_import}
"""
    )

    # import
    test_diopi_manual_import = CodeTemplate(
        r"""
from conformance.diopi_manual_test import ManualTest
"""
    )

    test_diopi_nhwc_import = CodeTemplate(
        r"""
from conformance.diopi_runtime import set_nhwc
"""
    )

    # marks
    test_function_case_dtype_marks = CodeTemplate(
        r"""
@pytest.mark.${dtype}
"""
    )

    # test_case
    test_function_templ = CodeTemplate(
        r"""
${test_dtype_marks}
def test_${func_case_name}(self):
    ${forward}
    ${backward}
    ${forward_inp}
    default_context.clear_tensors()
"""
    )

    test_function_body_forward = CodeTemplate(
        r"""
f_in = os.path.join(data_path, '${test_module_name}', 'inputs', '${input_data_path}')
${test_function_ref_data_path}

# read input from file
with open(f_in, 'rb') as f:
    data_in = pickle.load(f)

function_paras = data_in['function_paras']
function_kwargs = function_paras['kwargs']
function_config = data_in['cfg']
ignore_paras_for_input_check = ${ignore_paras_for_input_check}
np_inputs_orign = deepcopy(function_kwargs)

${preprocess_parameters}

# output of device: dev_foward_out
${test_function_forward_call}
"""
    )

    test_preprocess_parameters = CodeTemplate(
        r"""
${set_four_bytes}${set_nhwc}${set_stride}${to_tensor}
"""
    )

    test_set_four_bytes = CodeTemplate(
        r"""
# set four bytes
for para_key, para_val in function_kwargs.items():
    if para_key in ${dtype_list} and para_val is not None and para_val.dtype == np.int64:
        function_kwargs[para_key] = para_val.astype(np.int32)
"""
    )

    test_set_nhwc = CodeTemplate(
        r"""
for para_key, para_val in function_kwargs.items():
    if isinstance(para_val, np.ndarray) and para_key in ${nhwc_list}:
        if para_val.ndim < ${nhwc_min_dim} or para_val.ndim > 5:
            default_context.clear_tensors()
            pytest.xfail(f"Skipped: {para_val.ndim}-dim Tensor skipped for nhwc test")
        function_kwargs[para_key] = set_nhwc(para_val, ${nhwc_list}[0])
"""
    )

    test_set_stride = CodeTemplate(
        r"""
# set stride
args_stride = ${args_stride}
for para_key, para_val in function_kwargs.items():
    if isinstance(para_val, np.ndarray) and para_key + "stride" in ${has_stride}:
        stride = args_stride[str(para_key) + "stride"]
        assert len(stride) == len(para_val.shape), "stride must have same dim with shape"
        sumsize = int(sum((s - 1) * st for s, st in zip(para_val.shape, stride)) + 1)
        stride_pre_para_val = np.empty(sumsize, para_val.dtype)
        stride_para_val = np.lib.stride_tricks.as_strided(stride_pre_para_val, shape=para_val.shape, strides=tuple(para_val.dtype.itemsize * st for st in stride))
        np.copyto(stride_para_val, para_val)
        para_val = stride_para_val
        function_kwargs[para_key] = para_val
"""
    )

    test_to_tensor = CodeTemplate(
        r"""
# to_tensor
for para_key, para_val in function_kwargs.items():
    if isinstance(para_val, np.ndarray):
        function_kwargs[para_key] = Tensor.from_numpy(para_val)
    if para_key == 'dtype':
        function_kwargs[para_key] = from_numpy_dtype(para_val)
"""
    )

    test_to_tensor_list = CodeTemplate(
        r"""
# to_tensor
for para_key, para_val in function_kwargs.items():
    if para_key in ${gen_policy_args}:
        for idx, ele in enumerate(function_kwargs[para_key]):
            function_kwargs[para_key][idx] = Tensor.from_numpy(ele)
    elif isinstance(para_val, np.ndarray):
        function_kwargs[para_key] = Tensor.from_numpy(para_val)
"""
    )

    test_diopi_function_forward_call = CodeTemplate(
        r"""
function_config['atol'] = ${test_atol}
function_config['rtol'] = ${test_rtol}
function_config['atol_half'] = ${test_atol_half}
function_config['rtol_half'] = ${test_rtol_half}
tol = ${test_compare_tol}
# sum_to_compare
sum_to_compare = True if 'sorted' in function_kwargs and ~function_kwargs['sorted'] else False
tol['sum_to_compare'] = sum_to_compare
try:
    dev_foward_out = ${test_diopi_func_name}(**function_kwargs)
except (FunctionNotImplementedError, FunctionNotDefinedError) as e:
    default_context.clear_tensors()
    pytest.xfail(str(e))

# read ref_foward_out
with open(f_out, 'rb') as f:
    ref_foward_out = pickle.load(f)

try:
    CheckResult.compare_input_dict(function_kwargs, np_inputs_orign, ignore_paras_for_input_check, **tol)
    CheckResult.compare_output(dev_foward_out, ref_foward_out, **tol)
except Exception as e:
    default_context.clear_tensors()
    assert False, f'Test {function_config["name"]}: {function_config} traceback: {e}'
"""
    )

    test_diopi_function_inp_forward_call = CodeTemplate(
        r"""
# inplace call for the function
${test_diopi_func_inp_remove_grad_args}
try:
    dev_inp_forward_out = ${test_diopi_func_name}(inplace=True, **function_kwargs)
except (FunctionNotImplementedError, FunctionNotDefinedError) as e:
    default_context.clear_tensors()
    pytest.xfail(str(e))

try:
    ignore_paras_for_input_check.add('input')
    CheckResult.compare_input_dict(function_kwargs, np_inputs_orign, ignore_paras_for_input_check, **tol)
    CheckResult.compare_output(dev_inp_forward_out, ref_foward_out, **tol)
except Exception as e:
    default_context.clear_tensors()
    assert False, f'Test {function_config["name"]}  inplace: {function_config} traceback: {e}'
"""
    )
    test_diopi_func_inp_remove_grad_args = CodeTemplate(
        r"""
function_kwargs = {key: value for key, value in function_kwargs.items() if key not in backward_para.keys()}
"""
    )

    test_manual_function_forward_call = CodeTemplate(
        r"""
try:
    ManualTest.test_${test_diopi_func_name}(**function_kwargs)
    CheckResult.compare_input_dict(function_kwargs, np_inputs_orign, ignore_paras_for_input_check)
except (FunctionNotImplementedError, FunctionNotDefinedError) as e:
    default_context.clear_tensors()
    pytest.xfail(str(e))
"""
    )
    test_manual_function_inp_forward_call = CodeTemplate(
        r"""
${test_diopi_func_inp_remove_grad_args}
function_kwargs.update({'inplace': True})
try:
    ManualTest.test_${test_diopi_func_name}(**function_kwargs)
except (FunctionNotImplementedError, FunctionNotDefinedError) as e:
    default_context.clear_tensors()
    pytest.xfail(str(e))
"""
    )

    # backward
    test_function_body_backward = CodeTemplate(
        r"""
# grad_output_path
f_bp_out = os.path.join(data_path, '${test_module_name}', 'outputs', '${bp_output_data_path}')

if not isinstance(dev_foward_out, (list, tuple)):
    dev_foward_out = [dev_foward_out]
requires_backward = function_config['requires_backward']
outputs_for_backward = dev_foward_out if len(requires_backward) == 0 \
    else [dev_foward_out[i] for i in requires_backward]
backward_para = {}
grad_outputs = [ones_like(i) for i in outputs_for_backward]
backward_para["grad_outputs"] = grad_outputs
for k, v in function_config['saved_args'].items():
    backward_para[k] = dev_foward_out[v]

backward_para_compare = [item for value in backward_para.values() for item in (value if isinstance(value, list) else [value])]
backward_para_origin = deepcopy([item.numpy() for item in backward_para_compare])
try:
    dev_bp_out = ${test_diopi_bp_func_name}(**function_kwargs, **backward_para)
except (FunctionNotImplementedError, FunctionNotDefinedError) as e:
    default_context.clear_tensors()
    pytest.xfail(str(e))

with open(f_bp_out, 'rb') as f:
    ref_bp_out = pickle.load(f)

# checkout
try:
    CheckResult.compare_input_dict(function_kwargs, np_inputs_orign, ignore_paras_for_input_check, **tol)
    CheckResult.compare_input_list(backward_para_compare, backward_para_origin, **tol)
    CheckResult.compare_output(dev_bp_out, ref_bp_out, **tol)
except Exception as e:
    default_context.clear_tensors()
    assert False, f'Test {function_config["name"]} backward: {function_config} traceback: {e}'
"""
    )
