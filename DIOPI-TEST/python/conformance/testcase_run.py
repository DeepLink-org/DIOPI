import os
import pickle
import numpy as np

from . import functions as F
from .utils import logger, FunctionNotImplementedError
from .litert import Tensor
from .gen_inputs import inputs_dir_path, outputs_dir_path, load_testcases
from .gen_outputs import grad_kv


def convert_input_tensors(function_paras: dict):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            function_paras['kwargs'][para] = Tensor.from_numpy(function_paras['kwargs'][para])


def allclose(cfg: dict, tensor1: np.ndarray, tensor2: np.ndarray) -> bool:
    rtol = cfg.get('rtol', 1e-5)
    atol = cfg.get('atol', 1e-8)
    return np.allclose(tensor1, tensor2, rtol, atol, True)


def compare(outputs, cfg, outputs_reference):
    passed = True
    if isinstance(outputs, Tensor):
        passed = allclose(cfg, outputs.numpy(), outputs_reference)
    elif isinstance(outputs, (list, tuple)):
        assert isinstance(outputs_reference, (list, tuple))
        assert len(outputs) == len(outputs_reference)
        for i in range(len(outputs)):
            if isinstance(outputs[i], Tensor):
                passed &= allclose(cfg, outputs[i].numpy(), outputs_reference[i])
            if ~passed:
                return False
    elif isinstance(outputs, dict):
        assert isinstance(outputs_reference, dict)
        assert len(outputs) == len(outputs_reference)
        for k, v in outputs.items():
            if isinstance(v, Tensor):
                passed = passed and allclose(cfg, v.numpy(), outputs_reference[k])
            if ~passed:
                return False
    return passed


def run(opname):
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)
            op_name = data["cfg"]["name"]
            if opname not in ['all', op_name]:
                continue

            fn_paras = data["function_paras"]
            convert_input_tensors(fn_paras)
            kwargs = fn_paras['kwargs']

            op_calls = []
            op_calls.append(f"F.{op_name}(**kwargs)")
            if data["cfg"].get("is_inplace", False):
                op_calls.append(f"F.{op_name}(**kwargs, inplace=True)")

            for op_call in op_calls:
                try:
                    outputs = eval(op_call)

                    with open(os.path.join(outputs_dir_path, fname), "rb") as file_outputs:
                        outputs_reference = pickle.load(file_outputs)
                        passed = compare(outputs, data['cfg'], outputs_reference)
                        if passed:
                            logger.info(f"run {op_name} succeed")
                        else:
                            logger.info(f"run {op_name} failed")
                except FunctionNotImplementedError as e:
                    logger.info(f"function {op_name} is not implemented, {e}")
                except Exception as e:
                    logger.info(f"run {op_name} failed with exception {e}")

            if "do_backward" in data["cfg"].keys():
                fname = fname.split(".pth")[0] + "_backward.pth"
                requires_backward = data["cfg"]["requires_backward"]
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                if len(requires_backward) == 0:
                    outputs_for_backward = outputs
                else:
                    outputs_for_backward = [outputs[i] for i in requires_backward]

                _, inputs_for_grad = grad_kv(fn_paras)
                saved_grads = data["cfg"]['saved_grads']
                backward_para = {}
                if len(inputs_for_grad) != 0:
                    grad_output = [F.ones_like(i) for i in outputs_for_backward]
                    backward_para["grad_output"] = grad_output
                    for k, v in saved_grads.items():
                        backward_para[k] = outputs[v]

                    fn_paras.update(backward_para)
                    kwargs = fn_paras['kwargs']
                    grad_inputs = eval(f"F.{op_name}_backward(**kwargs)")
                    with open(os.path.join(outputs_dir_path, fname), "rb") as file_outputs:
                        outputs_reference = pickle.load(file_outputs)
                        passed = compare(grad_inputs, data['cfg'], outputs_reference)
                        if passed:
                            logger.info(f"run {op_name} Backward succeed")
                        else:
                            logger.info(f"run {op_name} Backward failed")
