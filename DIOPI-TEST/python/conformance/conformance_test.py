import os
import pickle
import numpy as np

from . import diopi_functions as F
from .utils import logger, FunctionNotImplementedError
from .utils import need_process_func
from .diopi_runtime import Tensor
from .gen_data import inputs_dir_path, outputs_dir_path
from .gen_data import get_saved_pth_list, get_data_from_file
from .utils import save_precision, record, write_precision

def convert_input_tensors(function_paras: dict):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            function_paras['kwargs'][para] = Tensor.from_numpy(function_paras['kwargs'][para])

        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                tensors[idx] = Tensor.from_numpy(ele)
            function_paras['kwargs'][para] = tensors


def allclose(cfg: dict, tensor1: np.ndarray, tensor2: np.ndarray, sum_to_compare=False, var_name="out") -> bool:
    rtol = cfg.get('rtol_half', 1e-5) if tensor1.dtype == np.float16 else cfg.get('rtol', 1e-5)
    atol = cfg.get('atol_half', 1e-8) if tensor1.dtype == np.float16 else cfg.get('atol', 1e-8)
    tensor1 = np.sum(tensor1) if sum_to_compare else tensor1
    tensor2 = np.sum(tensor2) if sum_to_compare else tensor2
    passed = np.allclose(tensor1, tensor2, rtol, atol, True)
    if record:
        save_precision(cfg, tensor1, tensor2, passed, var_name)
    return passed


def compare_with_gen_output(output, cfg, output_reference, sum_to_compare=False):
    passed = True
    if isinstance(output, Tensor):
        passed = allclose(cfg, output.numpy(), output_reference, sum_to_compare)
    elif isinstance(output, (list, tuple)):
        assert isinstance(output_reference, (list, tuple))
        assert len(output) == len(output_reference)
        for i in range(len(output)):
            if isinstance(output[i], Tensor):
                passed &= allclose(cfg, output[i].numpy(), output_reference[i], sum_to_compare, "out" + str(i))
            if not record and not passed:
                return False 
    elif isinstance(output, dict):
        assert isinstance(output_reference, dict)
        assert len(output) == len(output_reference)
        for k, v in output.items():
            if isinstance(v, Tensor):
                passed = passed and allclose(cfg, v.numpy(), output_reference[k], False, k)
            if not record and not passed:
                return False 
    elif isinstance(output, (int, float)):
        assert isinstance(output_reference, np.ndarray), "output_reference should be type numpy.array"
        output = np.array(output)
        assert output.shape == output_reference.shape, "output and output_reference should be same shape"
        passed = passed and allclose(cfg, output, output_reference, False, "scalar")
    else:
        return False

    return passed


class ManualTest(object):
    def test_dropout(input, p=0.5, training=True, inplace=False):
        input_numpy = input.numpy()
        out, mask = F.dropout(input, p, training, inplace)
        out_numpy = out.numpy()
        mask_numpy = mask.numpy()

        # compute ratio
        real_ratio = np.sum(mask_numpy) / mask.numel()
        # check data
        remains = out_numpy[mask_numpy == 1]
        ref = input_numpy[mask_numpy == 1]

        assert np.allclose(remains, ref / (1 - p), rtol=1e-4, atol=1e-5),\
            "failed to execute dropout"

        assert np.abs(real_ratio - (1 - p)) < 3e-2,\
            "failed to execute dropout"

    def test_randperm(n):
        out = F.randperm(n)
        out_numpy = out.numpy()
        out_ref = np.arange(0, n, 1)
        if out.numel() > 10:
            assert not np.allclose(out_numpy, out_ref, 1e-3),\
                "failed to execute randperm"

        out_numpy.sort()
        assert np.allclose(out_numpy, out_ref, 1e-3),\
            "failed to execute randperm"

    def test_uniform(input, start, end):
        out = F.uniform(input, start, end)
        out_numpy = out.numpy()

        assert (out_numpy <= end).all() and (out_numpy >= start).all(),\
            "failed to execute uniform"
        if out.numel() > 100:
            assert abs(out_numpy.mean() - (end + start)/2) < 1e-1,\
                "failed to execute uniform"

    def test_bernoulli(input, inplace=False, p=None):
        p_numpy = input.numpy()
        p = p_numpy.mean() if p is None else p
        out = F.bernoulli(input, inplace, p)
        out_numpy = out.numpy()

        if out.numel() > 100:
            assert abs(out_numpy.mean() - p) < 1e-1,\
                "failed to execute bernoulli"

    def test_random(input, start, end):
        out = F.random(input, start, end)
        out_numpy = out.numpy()

        assert (out_numpy >= start).all(),\
            "failed to execute random"
        if end is not None:
            assert (out_numpy <= end - 1).all(),\
                "failed to execute random"


class ConformanceTest(object):
    r'''
    Run all functions by using input, then compare_with_gen_output with saved output
    '''
    @staticmethod
    def run(func_name, model_name, filter_dtype_str_list):
        saved_pth_list = get_saved_pth_list()
        for saved_pth in saved_pth_list:
            cfg_func_name = saved_pth.split("::")[1].rsplit("_", 1)[0]
            if not need_process_func(cfg_func_name, func_name, model_name):
                continue

            input_abs_path = os.path.join(inputs_dir_path, saved_pth)
            output_abs_path = os.path.join(outputs_dir_path, saved_pth)
            data = get_data_from_file(input_abs_path, saved_pth, "input")
            if data is None:
                continue

            need_output = False if "no_output_ref" in data['cfg'] else True
            module = "F" if need_output else "ManualTest"
            test_func_name = cfg_func_name if need_output else "test_" + cfg_func_name
            if need_output:
                output_reference = get_data_from_file(output_abs_path, saved_pth, "output")
                if output_reference is None:
                    continue

            function_paras = data["function_paras"]
            convert_input_tensors(function_paras)
            kwargs = function_paras['kwargs']
            func_call_list = []
            func_call_list.append(f"{module}.{test_func_name}(**kwargs)")
            if data["cfg"].get("is_inplace", False):
                func_call_list.append(f"{module}.{test_func_name}(**kwargs, inplace=True)")

            for func_call in func_call_list:
                try:
                    output = eval(func_call)
                    sum_to_compare = True if 'sorted' in kwargs and ~kwargs['sorted'] else False
                    passed = compare_with_gen_output(output, data['cfg'], output_reference, sum_to_compare) \
                        if need_output else True
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

                write_precision(data["cfg"], cfg_func_name, passed)
        
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
                        write_precision(data["cfg"], cfg_func_name + '_bp', passed)
                    except FunctionNotImplementedError as e:
                        logger.error(f"NotImplemented: {e}")
                    except AttributeError as e:
                        logger.error(f"AttributeError: {e}")
                    except Exception as e:
                        logger.error(f"Failed: {e}")
