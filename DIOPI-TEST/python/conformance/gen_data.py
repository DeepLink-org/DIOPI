import os
import sys
import copy
import pickle
import numpy as np

from . import to_numpy_dtype
from .utils import logger
from .config import Genfunc, dict_elem_length, Config
from . import diopi_configs


_cur_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir_path = os.path.join(_cur_dir, "../data/inputs")
outputs_dir_path = os.path.join(_cur_dir, "../data/outputs")
cfg_file_name = "test_config.cfg"


def expand_para(para_dict: dict, paras_list: list):
    r'''
    dict(a = [1,2], b = [11,22])
    --->
    [dict(a = 1, b = 11), dict(a=2, b=22)]
    '''
    length = dict_elem_length(para_dict)
    for i in range(length):
        tmp_para_dict = {}
        for k, v in para_dict.items():
            tmp_para_dict[k] = copy.deepcopy(v[i])
        paras_list.append(tmp_para_dict)


def expand_tensor_para(args_list, tensor_paras_list):
    r'''
    [
        dict(
            "ins" = ["input", "weight"],
            "requires_grad" = [True, False],
            "dtype" = [Dtype.float32, Dtype.float64],
            "shape"=[(),(2,3)]
        )
    ]
    --->
    [
        [dict(
            "ins" = "input",
            "requires_grad" = True,
            "dtype" = [Dtype.float32, Dtype.float64],
            "shape"=[()]
        )
        dict(
            "ins" = "weight",
            "requires_grad" = False,
            "dtype" = [Dtype.float32, Dtype.float64],
            "shape"=[()]
        )],

       [dict(
            "ins" = "input",
            "requires_grad" = True,
            "dtype" = [Dtype.float32, Dtype.float64],
            "shape"=[(2,3)]
        )
        dict(
            "ins" = "weight",
            "requires_grad" = False,
            "dtype" = [Dtype.float32, Dtype.float64],
            "shape"=[(2,3)]
        )],
    ]
    '''
    if len(args_list) == 0 or args_list is None:
        return
    # expand ins, requires_grad
    tmp_args_list = []
    for arg in args_list:
        for i in range(len(arg["ins"])):
            tmp_arg = copy.deepcopy(arg)
            tmp_arg["ins"] = copy.deepcopy(arg["ins"][i])
            # tmp_arg["requires_grad"] = copy.deepcopy(
            #     args_list[i_obj]["requires_grad"][i])
            tmp_args_list.append(tmp_arg)

    args0_dict = tmp_args_list[0]
    assert "shape" in args0_dict or "value" in args0_dict
    num = len(args0_dict["shape"]) if "shape" in args0_dict else len(args0_dict["value"])

    for j in range(num):
        args_ins_expand_list = copy.deepcopy(tmp_args_list)
        for i in range(len(tmp_args_list)):
            if "value" in tmp_args_list[i].keys():
                args_ins_expand_list[i]["value"] = copy.deepcopy(
                    tmp_args_list[i]["value"][j])
            elif "shape" in tmp_args_list[i].keys():
                args_ins_expand_list[i]["shape"] = copy.deepcopy(
                    tmp_args_list[i]["shape"][j])
        tensor_paras_list.append(args_ins_expand_list)


def expand_cfg_by_para(cfg_dict: dict):
    paras_list = []
    tensor_paras_list = []

    expand_para(cfg_dict["para"], paras_list)
    expand_tensor_para(cfg_dict["tensor_para"]["args"], tensor_paras_list)
    return paras_list, tensor_paras_list


def expand_cfg_all(paras_list, tensor_paras_list, cfg_dict) -> list:
    cfg_expand_list = []
    if len(tensor_paras_list) != 0:
        arg_dtype_num = 0
        for arg in cfg_dict["tensor_para"]["args"]:
            if arg.get("dtype") is not None:
                arg_dtype_num = len(arg["dtype"])
                break

        if arg_dtype_num != 0:
            for i in range(arg_dtype_num):
                for j in range(len(tensor_paras_list)):
                    tmp_cfg_dict = copy.deepcopy(cfg_dict)
                    tmp_cfg_dict["tensor_para"]["args"] = copy.deepcopy(
                        tensor_paras_list[j])
                    if len(paras_list) != 0:
                        tmp_cfg_dict["para"] = copy.deepcopy(
                            paras_list[j])
                    for arg in tmp_cfg_dict["tensor_para"]["args"]:
                        if arg.get("dtype") is not None:
                            arg["dtype"] = copy.deepcopy(arg["dtype"][i])
                    cfg_expand_list.append(tmp_cfg_dict)
        # dtype does not exit in args, so do not take dtype into account
        else:
            for i in range(len(tensor_paras_list)):
                tmp_cfg_dict = copy.deepcopy(cfg_dict)
                tmp_cfg_dict["para"] = copy.deepcopy(paras_list[i])
                tmp_cfg_dict["tensor_para"]["args"] = copy.deepcopy(
                    tensor_paras_list[i])
                cfg_expand_list.append(tmp_cfg_dict)
    return cfg_expand_list


def expand_cfg_by_all_options(cfg_dict: dict) -> list:
    paras_list, tensor_paras_list = expand_cfg_by_para(cfg_dict)
    cfg_expand_list = expand_cfg_all(paras_list, tensor_paras_list, cfg_dict)
    return cfg_expand_list


def delete_if_gen_fn_in_tensor_para(cfg_dict):
    for arg in cfg_dict["tensor_para"]["args"]:
        if "gen_fn" in arg.keys():
            arg.pop("gen_fn")


def delete_fn(cfg_dict):
    for arg in cfg_dict["tensor_para"]["args"]:
        if "gen_fn" in arg.keys():
            arg.pop("gen_fn")
    return cfg_dict


def gen_tensor(arg: dict) -> np.ndarray:
    if "value" in arg.keys():
        dtype = to_numpy_dtype(arg.get("dtype", None))
        value = np.array(arg["value"], dtype=dtype)
        return value

    if arg["shape"] is None:
        return None

    try:
        shape = arg["shape"]
        if isinstance(arg["gen_fn"], int):
            gen_fn = arg["gen_fn"]
        else:
            gen_fn = arg["gen_fn"]["fn"]
            assert(gen_fn == Genfunc.randint), "only randint needs args"
            low = arg["gen_fn"].get("low", 0)
            high = arg["gen_fn"].get("high", 10)
        dtype = to_numpy_dtype(arg["dtype"])

        if gen_fn == Genfunc.randn:
            value = np.random.randn(*shape).astype(dtype)
        elif gen_fn == Genfunc.rand:
            value = np.random.rand(*shape).astype(dtype)
        elif gen_fn == Genfunc.ones:
            value = np.ones(shape, dtype=dtype)
        elif gen_fn == Genfunc.zeros:
            value = np.zeros(shape, dtype=dtype)
        elif gen_fn == Genfunc.mask:
            value = np.random.randint(low=0, high=2, size=shape).astype(dtype)
        elif gen_fn == Genfunc.randint:
            value = np.random.randint(low=low, high=high, size=shape, dtype=dtype)
        elif gen_fn == Genfunc.empty:
            value = np.empty(shape, dtype=dtype)
        else:
            value = np.random.randn(*shape).astype(dtype)

    except BaseException as e:
        logger.error(e, exc_info=True)
        logger.error(arg)
        sys.exit()

    return value


def gen_and_dump_data(dir_path: str, cfg_name: str, cfg_expand_list: list, cfg_save_dict: dict):
    construct_paras = {}
    function_paras = {"kwargs": {}, "requires_grad": {}}
    tensor_list = []
    i = 0

    for cfg_dict in cfg_expand_list:
        para_dict = cfg_dict["para"]
        tensor_para_args_list = cfg_dict["tensor_para"]["args"]

        for k in para_dict:
            construct_paras[k] = para_dict[k]

        for arg in tensor_para_args_list:
            name = arg["ins"]
            # length of gen_num_range must be 2, otherwise ignore gen_num_range
            if len(arg["gen_num_range"]) != 2:
                value = gen_tensor(arg)
                function_paras["kwargs"][name] = value
                function_paras["requires_grad"][name] = arg["requires_grad"]
            else:
                tensors_num = np.random.randint(arg['gen_num_range'][0],
                                                arg['gen_num_range'][1])
                arg.setdefault("tensors_num", tensors_num)
                for _ in range(tensors_num):
                    value = gen_tensor(arg)
                    tensor_list.append(value)
                assert(cfg_dict["tensor_para"]["seq_name"] != ""), "need a name the list of tensors"
        # tie all the function_paras in a list named seq_name
        if cfg_dict["tensor_para"]["seq_name"] != "":
            name = cfg_dict["tensor_para"]["seq_name"]
            new_list = []
            for j in tensor_list:
                new_list.append(j)
            for j in function_paras["kwargs"].values():
                new_list.append(j)
            function_paras["kwargs"][name] = new_list

        delete_if_gen_fn_in_tensor_para(cfg_dict)
        # cfg_dict = delete_fn(copy.deepcopy(cfg_dict))

        file_name = f"{cfg_name}_{i}.pth"
        i += 1
        cfg_save_dict[file_name] = cfg_dict
        function_paras["kwargs"].update(construct_paras)
        cfg_info = {"function_paras": function_paras, "cfg": cfg_dict}
        with open(os.path.join(dir_path, file_name), "wb") as f:
            pickle.dump(cfg_info, f)

        tensor_list = []
        function_paras['kwargs'] = {}
        function_paras["requires_grad"] = {}


def get_saved_pth_list() -> list:
    with open(os.path.join(inputs_dir_path, cfg_file_name), "rb") as f:
        cfg_dict = pickle.load(f)

    return [k for k in cfg_dict]


class GenInputData(object):
    r'''
    Generate input data for all functions by using diopi_configs
    '''

    @staticmethod
    def run(func_name):
        if not os.path.exists(inputs_dir_path):
            os.makedirs(inputs_dir_path)

        configs = Config.process_configs(diopi_configs)

        cfg_counter = 0
        cfg_save_dict = {}
        for cfg_name in configs:
            cfg_func_name = configs[cfg_name]["name"]
            if func_name not in ['all', cfg_func_name]:
                continue
            logger.info(f"Generate benchmark input data for diopi_functions.{cfg_func_name}")
            cfg_expand_list = expand_cfg_by_all_options(configs[cfg_name])
            cfg_counter += len(cfg_expand_list)
            gen_and_dump_data(inputs_dir_path, cfg_name, cfg_expand_list, cfg_save_dict)

        with open(os.path.join(inputs_dir_path, cfg_file_name), "wb") as f:
            pickle.dump(cfg_save_dict, f)

        logger.info(f"Generate test cases number: {cfg_counter}")
        logger.info("Generate benchmark input data done!")


class CustomizedTest(object):
    def slice_op(input, dim, index):
        import torch
        size = len(input.size())
        slice_args = [True for i in range(size)]
        slice_args[dim] = index
        return torch.Tensor.__getitem__(input, slice_args)

    def index(input, **kwargs):
        import torch
        new_args = []
        for ele in kwargs.values():
            if ele is None:
                hasEllipsis = True
                if hasEllipsis and Ellipsis not in new_args:
                    new_args.append(...)
            else:
                new_args.append(ele)
        return torch.Tensor.__getitem__(input, new_args)

    def sgd(param, param_grad, buf, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        import torch
        param.requires_grad = True
        param.grad = param_grad
        optimizer = torch.optim.SGD([param, ], lr, momentum, dampening, weight_decay, nesterov)
        optimizer.state[param]['momentum_buffer'] = buf
        optimizer.step()
        return param, buf


def transfer_tensor_to_device(function_paras: dict):
    import torch
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            tensor = torch.from_numpy(function_paras['kwargs'][para])
            if function_paras["requires_grad"][para] == [True]:
                tensor.requires_grad = True
            function_paras['kwargs'][para] = tensor.cuda()

        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                tensors[idx] = torch.from_numpy(ele).cuda()
            function_paras['kwargs'][para] = tensors


def get_name_and_data_for_grad(function_paras):
    inputs_for_grad_value = []
    inputs_for_grad_key = []
    for k, v in function_paras["kwargs"].items():
        if k in function_paras["requires_grad"].keys() \
                and function_paras["requires_grad"][k] == [True]:
            inputs_for_grad_value.append(v)
            inputs_for_grad_key.append(k)

    return inputs_for_grad_key, inputs_for_grad_value


def to_numpy(tensors):
    import torch
    if isinstance(tensors, torch.Tensor):
        ndarrays = tensors.detach().cpu().numpy()
    elif isinstance(tensors, (list, tuple)):
        ndarrays = []
        for i in range(len(tensors)):
            if isinstance(tensors[i], torch.Tensor):
                ndarrays.append(tensors[i].detach().cpu().numpy())
            else:
                ndarrays.append(tensors[i])
    elif isinstance(tensors, dict):
        ndarrays = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                ndarrays.update(k=v.detach().cpu().numpy())
            else:
                ndarrays.update(k=v)
    else:
        ndarrays = None

    return ndarrays


class GenOutputData(object):
    r'''
    Generate output data for all functions by using torch and input data
    '''

    @staticmethod
    def run(func_name):
        import torch
        import torchvision
        if not os.path.exists(inputs_dir_path):
            logger.error("Input data is not generated!")
            sys.exit(0)

        if not os.path.exists(outputs_dir_path):
            os.makedirs(outputs_dir_path)

        gen_counter = 0
        func_name_list = []  # make the info log once
        saved_pth_list = get_saved_pth_list()
        for saved_pth in saved_pth_list:
            input_abs_path = os.path.join(inputs_dir_path, saved_pth)
            if not os.path.exists(input_abs_path):
                logger.error(f"FileNotFound: No benchmark input data '{saved_pth}' was generated " \
                    f"(No such file or directory: {input_abs_path})")
                continue
            try:
                f = open(input_abs_path, "rb")
                data = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
            else:
                f.close()

            cfg_func_name = data["cfg"]["name"]
            if func_name not in ['all', cfg_func_name]:
                continue
            logger_str = "output"

            module = "torch.nn.functional"
            if "interface" in data["cfg"].keys():
                module = data["cfg"]["interface"][0]

            function_paras = data["function_paras"]
            transfer_tensor_to_device(function_paras)
            kwargs = function_paras['kwargs']
            func_call = f"{module}.{cfg_func_name}(**kwargs)"

            try:
                outputs = eval(func_call)
            except Exception as e:
                logger.error(f"Failed to execute function {func_call}, caused by {e}")
                continue

            if "do_backward" in data["cfg"].keys():
                saved_pth = saved_pth.split(".pth")[0] + "_backward.pth"
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]

                requires_backward = data["cfg"]["requires_backward"]
                outputs_for_backward = outputs if len(requires_backward) == 0 \
                    else [outputs[i] for i in requires_backward]

                inputs_name_for_grad, inputs_for_grad = get_name_and_data_for_grad(function_paras)
                saved_grads = None
                if len(inputs_for_grad) != 0:
                    grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                    grads = torch.autograd.grad(
                        outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                    saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}

                with open(os.path.join(outputs_dir_path, saved_pth), "wb") as f:
                    pickle.dump(to_numpy(saved_grads), f)

                logger_str = f"{logger_str} and backward"

            if cfg_func_name not in func_name_list:
                func_signature = f"diopi_functions.{cfg_func_name}"
                logger.info(f"Generate benchmark {logger_str} data for {func_signature}")
                func_name_list.append(cfg_func_name)

            if outputs is not None:
                with open(os.path.join(outputs_dir_path, saved_pth), "wb") as f:
                    pickle.dump(to_numpy(outputs), f)
                    gen_counter += 1

        logger.info(f"Generate test cases number: {gen_counter}")
        logger.info("Generate benchmark output and backward data done!")
