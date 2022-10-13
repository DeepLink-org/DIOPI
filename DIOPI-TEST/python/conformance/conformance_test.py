import os
import pickle
import numpy as np

from . import diopi_functions as F
from .utils import logger, FunctionNotImplementedError
from .diopi_runtime import Tensor
from .gen_data import inputs_dir_path, outputs_dir_path
from .gen_data import get_saved_pth_list, get_data_from_file


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
    rtol = cfg.get('rtol_half', 1e-5) if tensor1.dtype == np.float16 else cfg.get('rtol', 1e-5)
    atol = cfg.get('atol_half', 1e-8) if tensor1.dtype == np.float16 else cfg.get('atol', 1e-8)
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
            if not passed:
                # print(k, "faild:", v.numpy()[0], output_reference[k][0])
                return False
    elif isinstance(output, (int, float)):
        assert isinstance(output_reference, np.ndarray), "output_reference should be type numpy.array"
        output = np.array(output)
        assert output.shape == output_reference.shape, "output and output_reference should be same shape"
        passed = passed and allclose(cfg, output, output_reference)
    else:
        return False

    return passed


class ManualTest(object):
    def test_dropout(input, p=0.5, training=True, inplace=False):
        input_numpy = input.numpy()
        out = F.dropout(input, p, training, inplace)
        out_numpy = out.numpy()

        # compute ratio
        real_ratio = np.sum(out_numpy == 0) / out.numel()

        # check data
        remains = out_numpy[out_numpy != 0]
        ref = input_numpy[out_numpy != 0]
        assert np.allclose(remains, ref / (1 - p), 1e-3),\
            "failed to execute dropout"

        assert np.abs(real_ratio - p) < 2e-2,\
            "failed to execute dropout"


class ConformanceTest(object):
    r'''
    Run all functions by using input, then compare_with_gen_output with saved output
    '''
    @staticmethod
    def run(func_name):
        saved_pth_list = get_saved_pth_list()
        for saved_pth in saved_pth_list:
            cfg_func_name = saved_pth.split("::")[1].rsplit("_", 1)[0]
            if func_name not in ['all', cfg_func_name]:
                continue

            input_abs_path = os.path.join(inputs_dir_path, saved_pth)
            output_abs_path = os.path.join(outputs_dir_path, saved_pth)
            data = get_data_from_file(input_abs_path, saved_pth, "input")
            if data is None:
                continue

            need_output = False if "no_output_ref" in data['cfg'] else True
            module = "F" if need_output else "ManualTest"
            if need_output:
                output_reference = get_data_from_file(output_abs_path, saved_pth, "output")
                if output_reference is None:
                    continue
            else:
                cfg_func_name = "test_" + cfg_func_name

            function_paras = data["function_paras"]
            convert_input_tensors(function_paras)
            kwargs = function_paras['kwargs']
            func_call_list = []
            func_call_list.append(f"{module}.{cfg_func_name}(**kwargs)")
            if data["cfg"].get("is_inplace", False):
                func_call_list.append(f"{module}.{cfg_func_name}(**kwargs, inplace=True)")

            for func_call in func_call_list:
                try:
                    output = eval(func_call)
                    passed = compare_with_gen_output(output, data['cfg'], output_reference) if need_output else True
                    logger.info(f"Run diopi_functions.{cfg_func_name} succeed") \
                        if passed else logger.error(f"Run diopi_functions.{cfg_func_name} failed")
                except FunctionNotImplementedError as e:
                    logger.error(f"NotImplemented: {e}")
                    continue
                except AttributeError as e:
                    logger.error(f"AttributeError: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue

                if function_paras["requires_grad"] and "inplace=True" not in func_call:
                    saved_backward_pth = saved_pth.split(".pth")[0] + "_backward.pth"
                    saved_backward_pth = os.path.join(outputs_dir_path, saved_backward_pth)
                    backward_out_reference = get_data_from_file(saved_backward_pth, saved_pth, "backward output")
                    if backward_out_reference is None:
                        continue
                    if not isinstance(output, (list, tuple)):
                        output = [output]

                    requires_backward = data["cfg"]["requires_backward"]
                    outputs_for_backward = output if len(requires_backward) == 0 \
                        else [output[i] for i in requires_backward]

                    backward_para = {}
                    grad_outputs = [F.ones_like(i) for i in outputs_for_backward]
                    backward_para["grad_outputs"] = grad_outputs
                    for k, v in data["cfg"]['saved_args'].items():
                        backward_para[k] = output[v]

                    try:
                        grad_input = eval(f"F.{cfg_func_name}_backward(**kwargs, **backward_para)")
                        # import pdb;pdb.set_trace()
                        passed = compare_with_gen_output(grad_input, data['cfg'], backward_out_reference)
                        logger.info(f"Run diopi_functions.{cfg_func_name}_backward succeed") \
                            if passed else logger.error(f"Run diopi_functions.{cfg_func_name}_backward failed")
                    except FunctionNotImplementedError as e:
                        logger.error(f"NotImplemented: {e}")
                    except AttributeError as e:
                        logger.error(f"AttributeError: {e}")
                    except Exception as e:
                        logger.error(f"Failed: {e}")
