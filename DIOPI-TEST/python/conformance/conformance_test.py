import os
import pickle
import numpy as np

from . import diopi_functions as F
from .utils import logger, FunctionNotImplementedError
from .diopi_rt import Tensor
from .gen_input import inputs_dir_path, outputs_dir_path, get_saved_pth_list
from .gen_output import get_name_and_data_for_grad


def convert_input_tensors(function_paras: dict):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            function_paras['kwargs'][para] = Tensor.from_numpy(function_paras['kwargs'][para])

        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                tensors[idx] = Tensor.from_numpy(ele)
            function_paras['kwargs'][para] = tensors


def allclose(cfg: dict, tensor1: np.ndarray, tensor2: np.ndarray) -> bool:
    rtol = cfg.get('rtol', 1e-5)
    atol = cfg.get('atol', 1e-8)
    return np.allclose(tensor1, tensor2, rtol, atol, True)


def compare_with_gen_output(output, cfg, output_reference):
    passed = True
    if isinstance(output, Tensor):
        passed = allclose(cfg, output.numpy(), output_reference)
    elif isinstance(output, (list, tuple)):
        assert isinstance(output_reference, (list, tuple))
        assert len(output) == len(output_reference)
        for i in range(len(output)):
            if isinstance(output[i], Tensor):
                passed &= allclose(cfg, output[i].numpy(), output_reference[i])
            if not passed:
                return False
    elif isinstance(output, dict):
        assert isinstance(output_reference, dict)
        assert len(output) == len(output_reference)
        for k, v in output.items():
            if isinstance(v, Tensor):
                passed = passed and allclose(cfg, v.numpy(), output_reference[k])
            if ~passed:
                return False
    else:
        return False
    return passed


class ConformanceTest(object):
    r'''
    Run all functions by using input, then compare_with_gen_output with saved output
    '''
    @staticmethod
    def run(func_name):
        saved_pth_list = get_saved_pth_list()
        for saved_pth in saved_pth_list:
            output = None
            with open(os.path.join(inputs_dir_path, saved_pth), "rb") as f:
                data = pickle.load(f)

            cfg_func_name = data["cfg"]["name"]
            if func_name not in ['all', cfg_func_name]:
                continue

            function_paras = data["function_paras"]
            convert_input_tensors(function_paras)
            kwargs = function_paras['kwargs']

            try:
                func = eval(f"F.{cfg_func_name}")
            except AttributeError as e:
                logger.error(f"function {cfg_func_name} 's python interface is not implemented, {e}")
                continue

            func_call_list = []
            func_call_list.append(f"F.{cfg_func_name}(**kwargs)")
            if data["cfg"].get("is_inplace", False):
                func_call_list.append(f"F.{cfg_func_name}(**kwargs, inplace=True)")

            for func_call in func_call_list:
                with open(os.path.join(outputs_dir_path, saved_pth), "rb") as f:
                    output_reference = pickle.load(f)

                try:
                    output = eval(func_call)
                    passed = compare_with_gen_output(output, data['cfg'], output_reference)
                    logger.info(f"run {cfg_func_name} succeed") \
                        if passed else logger.error(f"run {cfg_func_name} failed")
                except FunctionNotImplementedError as e:
                    logger.error(f"function {cfg_func_name} is not implemented, {e}")
                except Exception as e:
                    logger.error(f"run {cfg_func_name} failed with exception: {e}")

            if "do_backward" in data["cfg"].keys() and output is not None:
                saved_pth = saved_pth.split(".pth")[0] + "_backward.pth"
                if not isinstance(output, (list, tuple)):
                    output = [output]

                requires_backward = data["cfg"]["requires_backward"]
                outputs_for_backward = output if len(requires_backward) == 0 \
                    else [output[i] for i in requires_backward]

                _, inputs_for_grad = get_name_and_data_for_grad(function_paras)
                backward_para = {}
                if len(inputs_for_grad) != 0:
                    grad_outputs = [F.ones_like(i) for i in outputs_for_backward]
                    backward_para["grad_outputs"] = grad_outputs
                    for k, v in data["cfg"]['saved_args'].items():
                        backward_para[k] = output[v]

                    function_paras.update(backward_para)
                    kwargs = function_paras['kwargs']
                    grad_input = eval(f"F.{cfg_func_name}_backward(**kwargs)")

                    with open(os.path.join(outputs_dir_path, saved_pth), "rb") as f:
                        output_reference = pickle.load(f)

                    passed = compare_with_gen_output(grad_input, data['cfg'], output_reference)
                    logger.info(f"run {cfg_func_name} backward succeed") \
                        if passed else logger.error(f"run {cfg_func_name} backward failed")
