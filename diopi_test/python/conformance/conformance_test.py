# Copyright (c) 2023, DeepLink.
import os
import numpy as np

from . import diopi_functions as F
from . import diopi_configs, ops_with_states
from .config import Config
from .utils import logger, FunctionNotImplementedError, DiopiException
from .utils import need_process_func, glob_vars, nhwc_op, dtype_op
from .diopi_runtime import Tensor, compute_nhwc_stride, default_context, diopi_rt_init, Generator
from diopilib import build_generator_state
from .utils import save_precision, record, write_precision
from .utils import get_saved_pth_list, get_data_from_file
from .utils import cfg_file_name
from .utils import default_cfg_dict
from . import model_config


def convert_input_tensors(function_paras: dict, test_tag: list, nhwc_list=[], dtype_list=[], filter_dtype_str_list=[]):
    tensor_info = []
    for para in function_paras["kwargs"].keys():
        tensor = function_paras['kwargs'][para]
        if glob_vars.four_bytes and (para in dtype_list) \
                and tensor is not None and tensor.dtype == np.int64:
            tensor = tensor.astype(np.int32)

        if isinstance(tensor, Tensor):
            tensor = tensor.numpy()
        if isinstance(tensor, np.ndarray):
            ndim = tensor.ndim
            if glob_vars.nhwc and (para in nhwc_list):
                if ndim < glob_vars.nhwc_min_dim or ndim > 5:
                    raise DiopiException(f"Skipped: {ndim}-dim Tensor skipped for nhwc test")
                tensor_nchw = tensor
                ndim = tensor_nchw.ndim
                if ndim == 3:
                    axis = (1, 2, 0)
                elif ndim == 4 and '3d' in nhwc_list:
                    axis = (1, 2, 3, 0)
                elif ndim == 4:
                    axis = (0, 2, 3, 1)
                elif ndim == 5:
                    axis = (0, 2, 3, 4, 1)
                tensor_nhwc = np.transpose(tensor_nchw, axis).copy()
                tensor_nhwc.shape = tensor_nchw.shape
                tensor_nhwc.strides = compute_nhwc_stride(tensor_nchw.shape, tensor_nchw.itemsize, nhwc_list[0])
                tensor = tensor_nhwc
                if 'nhwc' not in test_tag:
                    test_tag.append('nhwc')
            # 处理有stride输入的tensor
            elif str(para) + "stride" in function_paras:
                stride = function_paras[para + "stride"]
                assert len(stride) == len(tensor.shape), "stride must have same dim with shape"
                sumsize = int(sum((s - 1) * st for s, st in zip(tensor.shape, stride)) + 1)
                stride_pre_tensor = np.empty(sumsize, tensor.dtype)
                stride_tensor = np.lib.stride_tricks.as_strided(stride_pre_tensor, shape=tensor.shape, strides=tuple(tensor.dtype.itemsize * st for st in stride))
                np.copyto(stride_tensor, tensor)
                tensor = stride_tensor
            function_paras['kwargs'][para] = Tensor.from_numpy(tensor)
            if filter_dtype_str_list and str(tensor.dtype) in filter_dtype_str_list:
                raise DiopiException(f"Skipped: {tensor.dtype} Tensor skipped for test")
            if tensor is not None and str(tensor.dtype) not in test_tag:
                test_tag.append(str(tensor.dtype))
            tensor_info.append((para, str(tensor.dtype), str(tensor.shape)))
        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                tensors[idx] = Tensor.from_numpy(ele)
                if ele is not None and str(ele.dtype) not in test_tag:
                    test_tag.append(str(ele.dtype))
            if filter_dtype_str_list and str(tensor.dtype) in filter_dtype_str_list:
                raise DiopiException(f"Skipped: {tensor.dtype} Tensor skipped for test")
            function_paras['kwargs'][para] = tensors
            tensor_info.append(("TensorList: " + para, str(tensors[0].get_dtype()), str(tensors[0].shape())))
    return tensor_info


def allclose(cfg: dict, tensor1: np.ndarray, tensor2: np.ndarray, sum_to_compare=False, var_name="out") -> bool:
    rtol = cfg.get('rtol_half', 1e-5) if tensor1.dtype == np.float16 else cfg.get('rtol', 1e-5)
    atol = cfg.get('atol_half', 1e-8) if tensor1.dtype == np.float16 else cfg.get('atol', 1e-8)
    tensor1 = np.sum(tensor1) if sum_to_compare else tensor1
    tensor2 = np.sum(tensor2) if sum_to_compare else tensor2
    matched = np.isclose(tensor1, tensor2, rtol, atol, True)
    mismatched_num = matched.size - np.sum(matched)
    passed = mismatched_num <= default_cfg_dict['default_option']['mismatch_ratio_threshold'] * matched.size
    if record:
        save_precision(cfg, tensor1, tensor2, passed, var_name)
    if not passed:
        sum1 = tensor1.sum()
        sum2 = tensor2.sum()
        mask = np.isclose(tensor1, tensor2, rtol, atol, True)
        count = np.count_nonzero(np.equal(mask, False))
        if tensor1.dtype == np.bool_:
            max_diff = 1
            logger.info(f"The count of elements that do not meet the accuracy requirement is {count}.")
            logger.info(f"Max of diff is {max_diff}.")
            logger.debug(f"Sum of {var_name} is {sum1}, Sum of {var_name}_ref is {sum2}, Max of diff is {max_diff}. \
                    \n" + f"{var_name} is {tensor1},\n{var_name}_ref is {tensor2},\nMask is {mask}\n")
        else:
            assert tensor1.size == tensor2.size, "tensor1 element num does not equal tensor2's."
            diff = np.abs(tensor1 - tensor2)
            max_diff = np.nanmax(diff)
            max_diff_index = np.unravel_index(np.nanargmax(diff), diff.shape)
            max_diff_elem = tensor1[max_diff_index]
            max_diff_elem_ref = tensor2[max_diff_index]
            logger.info(f"The count of elements that do not meet the accuracy requirement is {count}.")
            logger.info(f"The dtype of {var_name} is {tensor1.dtype}.")
            logger.info(f"The shape of {var_name} is {tensor1.shape}.")
            logger.info(f"The stride of {var_name} is {np.divide(tensor1.strides, tensor1.itemsize).astype(np.int32)}.")
            logger.info(
                f"The max of diff is {max_diff}. Specifically, the actual val is {max_diff_elem} and the expected is {max_diff_elem_ref}.")
            logger.debug(f"Sum of {var_name} is {sum1}, Sum of {var_name}_ref is {sum2}, Max of diff is {max_diff}. \
                    \n" + f"{var_name} is {tensor1},\n{var_name}_ref is {tensor2},\nMask is {mask}\n")
            logger.debug(f"The dtype of {var_name} is {tensor1.dtype}.")
            logger.debug(f"The shape of {var_name} is {tensor1.shape}.")
            logger.debug(
                f"The stride of {var_name} is {np.divide(tensor1.strides, tensor1.itemsize).astype(np.int32)}.")
            logger.debug(
                f"The max of diff is {max_diff}. Specifically, the actual val is {max_diff_elem} and the expected is {max_diff_elem_ref}.\n")
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
    def test_dropout_(func, input, p=0.5, training=True, inplace=False):
        input_numpy = input.numpy()
        state = build_generator_state(input.context())
        generator = Generator(state)
        out, mask = func(input, p, training, inplace, generator)
        name = 'dropout' if func == F.dropout else 'dropout2d'
        out_numpy = out.numpy()
        mask_numpy = mask.numpy()

        rtol = 1e-2 if input_numpy.dtype == np.float16 else 1e-4
        atol = 5e-2 if input_numpy.dtype == np.float16 else 1e-5

        if training and input.numel() > 0:
            # compute ratio
            real_ratio = np.sum(mask_numpy) / mask.numel()
            # check data
            if func == F.dropout2d:
                tmp = np.ones(input.shape().data)
                mask_numpy = mask_numpy * tmp
            remains = out_numpy[mask_numpy == 1]
            ref = input_numpy[mask_numpy == 1]
            assert np.allclose(remains, ref / (1 - p), rtol=rtol, atol=atol),\
                f"failed to execute {name}, dropout value doesn't matches."
            if mask.numel() > 100:
                # 0.05 is from pytorch
                assert np.abs(real_ratio - (1 - p)) < 0.05,\
                    f"failed to execute {name}, dropout proportion unexpected."
        else:
            assert np.allclose(input_numpy, out_numpy, rtol=rtol, atol=atol),\
                f"failed to execute {name}, dropout value should be the same."

    def test_dropout(input, p=0.5, training=True, inplace=False):
        ManualTest.test_dropout_(F.dropout, input, p, training, inplace)

    def test_dropout2d(input, p=0.5, training=True, inplace=False):
        ManualTest.test_dropout_(F.dropout2d, input, p, training, inplace)

    def test_randperm(n):
        state = build_generator_state(default_context)
        generator = Generator(state)
        out = F.randperm(n, generator=generator)
        out_numpy = out.numpy()
        out_ref = np.arange(0, n, 1)
        if out.numel() > 10:
            assert not np.allclose(out_numpy, out_ref, 1e-3),\
                "failed to execute randperm"

        out_numpy.sort()
        assert np.allclose(out_numpy, out_ref, 1e-3),\
            "failed to execute randperm"

    def test_uniform(input, start=0, end=1):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.uniform(input, start, end, generator)
        epsilon = 1e-5   # eliminate minor precision error
        out_numpy = out.numpy()
        assert (out_numpy <= (end + epsilon)).all() and (out_numpy >= (start - epsilon)).all(),\
            "failed to execute uniform"
        if out.numel() > 100:
            assert abs(out_numpy.mean() - (end + start) / 2) < 1e-1,\
                "failed to execute uniform"

    def test_bernoulli(input, inplace=False, p=None):
        p_numpy = input.numpy()
        if input.numel() > 0:
            p = p_numpy.mean() if p is None else p
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.bernoulli(input, inplace, p, generator)
        out_numpy = out.numpy()

        assert np.all((out_numpy == 0) | (out_numpy == 1)), "bernoulli output must be 0 or 1"
        if out.numel() > 100:
            assert abs(out_numpy.mean() - p) < 1e-1,\
                "failed to execute bernoulli"

    def test_random(input, start, end):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.random(input, start, end, generator)
        out_numpy = out.numpy()

        assert (out_numpy >= start).all(),\
            "failed to execute random"
        if end is not None:
            assert (out_numpy <= end - 1).all(),\
                "failed to execute random"

    def test_randn(size):
        from scipy import stats
        out = F.randn(size)
        out_numpy = out.numpy().flatten()
        p_value = stats.kstest(out_numpy, 'norm', args=(0.0, 1.))[1]
        # pytorch uses 0.0001
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal, p_value is {p_value}"

    def test_normal(mean, std, size=None):
        from scipy import stats
        state = build_generator_state(default_context)
        generator = Generator(state)
        out = F.normal(mean, std, size, generator)
        out_numpy = out.numpy()
        if isinstance(mean, Tensor):
            mean_numpy = mean.numpy()
            out_numpy -= mean_numpy
            mean = 0.0
        if isinstance(std, Tensor):
            out_numpy -= mean
            std_numpy = std.numpy()
            out_numpy /= std_numpy
            mean = 0.0
            std = 1.
        out_numpy = out_numpy.flatten()
        if len(out_numpy) == 0:
            return True
        p_value = stats.kstest(out_numpy, 'norm', args=(mean, std + 1e-22))[1]
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal, p_value is {p_value}"

    def test_normal_(input, mean, std, shape=None):
        from scipy import stats
        input_size = 0 in input.size().data
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.normal_(input, mean, std, shape, generator)
        out_numpy = out.numpy()
        out_numpy = out_numpy.flatten()
        if len(out_numpy) == 0 and input_size:
            return True
        p_value = stats.kstest(out_numpy, 'norm', args=(mean, std))[1]
        assert p_value > 0.0001, f"can't pass the ks test, failed to execute normal_, p_value is {p_value}, shape of out is: {out_numpy.shape}"

    def test_multinomial(input, num_samples, replacement=False):
        state = build_generator_state(input.context())
        generator = Generator(state)
        out = F.multinomial(input, num_samples, replacement, generator)
        out_numpy = out.numpy()
        has_duplicates = False
        if out.size().len == 2:
            has_duplicates = len(out_numpy[0]) != len(set(out_numpy[0]))
        else:
            has_duplicates = len(out_numpy) != len(set(out_numpy))
        if not replacement:
            assert has_duplicates is False, "failed to execute multinomial"
        out_numpy = out_numpy.flatten()
        assert len(out_numpy) % num_samples == 0, "failed to execute multinomial"


def config_to_format_string(data, indent=0):
    yaml_str = ""
    if isinstance(data, dict):
        for key, value in data.items():
            if value is None or value == [] or value == {} or value == "":
                continue
            yaml_str += "\n" + " " * indent + f"{key}: "
            if key not in ["shape", "value"]:
                yaml_str += config_to_format_string(value, indent + 2)
            else:
                yaml_str += config_to_format_string(str(value), indent + 2)
    elif isinstance(data, (list, tuple)):
        for item in data:
            yaml_str += "\n" + " " * indent + "- " + config_to_format_string(item, indent + 2)
    else:
        yaml_str += f"{data}"
    return yaml_str


def check_device_para_and_tensor_para(cfg_dicts, device_cfg_dicts, cfg_name):
    cfg_dict = cfg_dicts[cfg_name]
    device_cfg_dict = device_cfg_dicts[cfg_name]
    para_dict = cfg_dict["para"]
    device_para_dict = device_cfg_dict["para"]
    for dk, dv in device_para_dict.items():
        if dk in para_dict:
            v = para_dict[dk]
            for x in dv:
                if x not in v:
                    logger.error(
                        f"Para {x} of key {dk} for {cfg_name} in device_configs not found in diopi_configs. Ignored.")

    args_list = cfg_dict["tensor_para"]["args"]
    device_tensor_paras_dict = device_cfg_dict["tensor_para"]["args"]
    for input in device_tensor_paras_dict.keys():
        in_found = False
        for args in args_list:
            if "ins" in args:
                ins = args["ins"]
                if input in ins:
                    in_found = True
                    for key in ["dtype", "shape", "value"]:
                        if key in device_tensor_paras_dict[input] and key in args:
                            for dv in device_tensor_paras_dict[input][key]:
                                if dv not in args[key]:
                                    logger.error(
                                        f"Tensor para {dv} of key {key} for {cfg_name} in device_configs not found in diopi_configs for ins {ins}. Ignored.")
        if not in_found:
            logger.error(f"Input name {input} for {cfg_name} in device_configs not found in diopi_configs. Ignored.")


def get_np_inputs(input_params: dict, ignore_params):
    np_inputs = {}
    for name, value in input_params.items():
        if name in ignore_params:
            continue
        if isinstance(value, np.ndarray):
            np_inputs[name] = value
        # transform tensor to numpy
        if isinstance(value, Tensor):
            np_inputs[name] = value.numpy()
    return np_inputs


def np_allclose(np_values1: dict, np_values2: dict):
    passed = True
    not_passed_name = ""
    for name, value in np_values1.items():
        assert name in np_values2.keys(), f"{name} not exist in np_values2"
        matched = np.isclose(value, np_values2[name], equal_nan=True)
        mismatched_num = matched.size - np.sum(matched)
        passed = mismatched_num <= default_cfg_dict['default_option']['mismatch_ratio_threshold'] * matched.size
        if not passed:
            not_passed_name = name
            break
    return passed, not_passed_name


class ConformanceTest(object):
    r'''
    Run all functions by using input, then compare_with_gen_output with saved output
    '''
    @staticmethod
    def run(func_name, model_name, filter_dtype_str_list, debug_level, impl_folder):

        diopi_rt_init()
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        inputs_dir_path = os.path.join(_cur_dir, "../data/" + model_name + "/inputs")
        outputs_dir_path = os.path.join(_cur_dir, "../data/" + model_name + "/outputs")

        saved_pth_list = get_saved_pth_list(inputs_dir_path, cfg_file_name)

        if model_name != "":
            diopi_config = "model_config." + model_name + "_config"
            configs = Config.process_configs(eval(diopi_config))
        else:
            configs = Config.process_configs(diopi_configs)

        use_device_configs = impl_folder != ''
        if use_device_configs:
            src_path = os.path.join(impl_folder, "device_configs.py")
            if os.path.isfile(src_path):
                dst_path = os.path.join(_cur_dir, "device_configs.py")

                def unlink_device_configs():
                    if os.path.islink(dst_path):
                        os.unlink(dst_path)
                unlink_device_configs()
                os.symlink(src_path, dst_path)
                import atexit
                atexit.register(unlink_device_configs)
                from .device_configs import device_configs
                from .device_config_helper import DeviceConfig
                device_configs = DeviceConfig.process_configs(device_configs)
                for cfg_name in configs:
                    cfg_func_name = configs[cfg_name]["name"]
                    if not need_process_func(cfg_func_name, func_name, model_name):
                        continue
                    if cfg_name in device_configs:
                        check_device_para_and_tensor_para(configs, device_configs, cfg_name)
            else:
                logger.error(f"device_configs.py not found in directory: {impl_folder} !")
                import sys
                sys.exit(0)

        for saved_pth in saved_pth_list:
            cfg_name = saved_pth.rsplit("_", 1)[0]
            cfg_func_name = saved_pth.split("::")[1].rsplit("_", 1)[0]
            if not need_process_func(cfg_func_name, func_name, model_name):
                continue

            input_abs_path = os.path.join(inputs_dir_path, saved_pth)
            output_abs_path = os.path.join(outputs_dir_path, saved_pth)
            data = get_data_from_file(input_abs_path, saved_pth, "input")
            if data is None:
                continue

            skipped = False
            if use_device_configs:
                if cfg_name in device_configs:
                    device_cfg = device_configs[cfg_name]

                    paras = data['cfg']['para']
                    device_paras = device_cfg['para']
                    for para_k, para_v in paras.items():
                        if para_k in device_paras and para_v in device_paras[para_k]:
                            skipped = True
                            break

                    tensor_paras = data['cfg']['tensor_para']['args']
                    device_tensor_paras = device_cfg['tensor_para']['args']
                    for tensor_para in tensor_paras:
                        if skipped:
                            break
                        if 'dtype' in device_cfg and tensor_para['dtype'] in device_cfg['dtype']:
                            skipped = True
                        if tensor_para['ins'] in device_tensor_paras:
                            device_tensor_para = device_tensor_paras[tensor_para['ins']]
                            need_checking_keys = ['dtype', 'value', 'shape']
                            for ck in need_checking_keys:
                                if ck in tensor_para and ck in device_tensor_para:
                                    if tensor_para[ck] in device_tensor_para[ck]:
                                        skipped = True

                    tol_keys_list = ['atol', 'rtol', 'atol_half', 'rtol_half']
                    for key in tol_keys_list:
                        if key in device_cfg:
                            data['cfg'][key] = device_cfg[key]

            if skipped:
                logger.warning(f"Run diopi_functions.{cfg_func_name} skipped")
                continue

            need_output = False if "no_output_ref" in data['cfg'] else True
            module = "F" if need_output else "ManualTest"
            test_func_name = cfg_func_name if need_output else "test_" + cfg_func_name
            if need_output:
                output_reference = get_data_from_file(output_abs_path, saved_pth, "output")
                if output_reference is None:
                    continue
            for index in range(len(data['cfg']['tensor_para']['args'])):
                para = data['cfg']['tensor_para']['args'][index]['ins']
                if str(para) + "stride" in data['cfg']['tensor_para']['args'][0].keys():
                    data['function_paras'][str(para) + "stride"] = data['cfg']['tensor_para']['args'][0][str(para) + "stride"]
            function_paras = data["function_paras"]
            test_tag = data["cfg"]["tag"]
            tensor_info = []
            nhwc_list = nhwc_op[cfg_func_name] if glob_vars.nhwc and (cfg_func_name in nhwc_op) else []
            dtype_list = dtype_op[cfg_func_name] if glob_vars.four_bytes and (cfg_func_name in dtype_op) else []
            kwargs = function_paras['kwargs']
            func_call_list = []
            func_call_list.append(f"{module}.{test_func_name}(**kwargs)")
            is_inplaces = []
            if "inplace" in kwargs.keys():
                is_inplaces.append(kwargs["inplace"])
            else:
                is_inplaces.append(False)
                if data["cfg"].get("is_inplace", False):
                    is_inplaces.append(True)
                    func_call_list.append(f"{module}.{test_func_name}(**kwargs, inplace=True)")

            ignore_paras_for_input_check = ops_with_states.get(test_func_name, set())
            for func_call, is_inplace in zip(func_call_list, is_inplaces):
                if is_inplace:
                    if test_tag and test_tag[-1] == 'backward':
                        test_tag.pop()
                    test_tag.append("inplace")
                try:
                    if is_inplace:
                        ignore_paras_for_input_check.add("input")
                    np_inputs_orign = get_np_inputs(function_paras['kwargs'], ignore_paras_for_input_check)
                    info = convert_input_tensors(function_paras, test_tag, nhwc_list, dtype_list, filter_dtype_str_list)
                    tensor_info = info if info else tensor_info
                    glob_vars.cur_test_func = func_call.split('(')[0].split('.')[1]
                    output = eval(func_call)
                    np_inputs_after_forward = get_np_inputs(function_paras['kwargs'], ignore_paras_for_input_check)
                    passed, not_passed_name = np_allclose(np_inputs_orign, np_inputs_after_forward)
                    sum_to_compare = True if 'sorted' in kwargs and ~kwargs['sorted'] else False
                    passed = passed and compare_with_gen_output(
                        output, data['cfg'], output_reference, sum_to_compare) if need_output else True
                    if passed:
                        logger.info(f"Run diopi_functions.{cfg_func_name} succeed")
                    else:
                        logger.info(output_abs_path)
                        logger.info(data['cfg'])
                        input_compare_str = "" if not_passed_name == "" else f", because of inputs: {not_passed_name} changed"
                        logger.error(
                            f"Run diopi_functions.{cfg_func_name} failed{input_compare_str}", tag=test_tag, info=tensor_info)
                        if debug_level > 0:
                            logger.error("failed config:\n%s", config_to_format_string(data['cfg']))
                            if debug_level > 1:
                                logger.error("failed arguments:")
                                for key, arg in kwargs.items():
                                    logger.error(f"{key}: {arg}")
                                logger.error(f"output_reference:\n{output_reference}")
                                logger.error(f"output:\n{output}")
                except FunctionNotImplementedError as e:
                    logger.error(f"NotImplemented: {e}")
                    continue
                except AttributeError as e:
                    logger.error(f"AttributeError: {e}")
                    continue
                except Exception as e:
                    logger.error(f"{e}")
                    continue

                write_precision(data["cfg"], cfg_func_name, passed)

                if function_paras["requires_grad"] and not is_inplace:
                    test_tag.append("backward")
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
                        np_inputs_after_backward = get_np_inputs(kwargs, ignore_paras_for_input_check)
                        passed, not_passed_name = np_allclose(np_inputs_orign, np_inputs_after_backward)
                        passed = passed and compare_with_gen_output(grad_input, data['cfg'], backward_out_reference)
                        if passed:
                            logger.info(f"Run diopi_functions.{cfg_func_name}_backward succeed")
                        else:
                            input_compare_str = "" if not_passed_name == "" else f", because of inputs: {not_passed_name} changed"
                            logger.error(
                                f"Run diopi_functions.{cfg_func_name}_backward failed{input_compare_str}", tag=test_tag, info=tensor_info)
                            if debug_level > 0:
                                logger.error("failed config:\n%s", config_to_format_string(data['cfg']))
                                if debug_level > 1:
                                    logger.error("failed arguments:")
                                    for key, arg in kwargs.items():
                                        logger.error(f"{key}: {arg}")
                                    for key, arg in backward_para.items():
                                        logger.error(f"{key}: {arg}")
                                    logger.error(f"grad_reference:\n{backward_out_reference}")
                                    logger.error(f"grad:\n{grad_input}")
                        write_precision(data["cfg"], cfg_func_name + '_bp', passed)
                    except FunctionNotImplementedError as e:
                        logger.error(f"NotImplemented: {e}")
                    except AttributeError as e:
                        logger.error(f"AttributeError: {e}")
                    except Exception as e:
                        logger.error(f"Failed: {e}")
            default_context.clear_tensors()
