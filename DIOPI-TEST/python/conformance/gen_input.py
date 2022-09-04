import configparser
import logging
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



def expand_para(para_dict: dict, key_idx: int, paras_list: list):
    r'''
    dict(a1=[1,2], a2=[11,22])
    ====>
    [
        dict(a1=1, a2=11),
        dict(a1=1, a2=22),
        dict(a1=2, a2=11),
        dict(a1=2, a2=22),
    ]
    '''
    if key_idx >= len(para_dict):
        paras_list.append(para_dict)
    else:
        keys = list(para_dict.keys())
        for i in para_dict[keys[key_idx]]:
            tmp_para_dict = copy.deepcopy(para_dict)
            tmp_para_dict[keys[key_idx]] = i
            expand_para(tmp_para_dict, key_idx + 1, paras_list)


def expand_related_para(para_dict: dict, related_paras_list: list):
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
        related_paras_list.append(tmp_para_dict)


def expand_call_para(args_list, call_paras_list):
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
        call_paras_list.append(args_ins_expand_list)


def expand_cfg_by_para(cfg_dict: dict):
    paras_list = []
    related_paras_list = []
    call_paras_list = []

    expand_para(cfg_dict["para"], 0, paras_list)
    expand_related_para(cfg_dict["related_para"], related_paras_list)
    expand_call_para(cfg_dict["call_para"]["args"], call_paras_list)
    return paras_list, related_paras_list, call_paras_list


def expand_cfg_all(paras_list, related_paras_list, call_paras_list, cfg_dict) -> list:
    cfg_expand_list = []
    for para in paras_list:
        if len(call_paras_list) != 0:
            arg_dtype_num = 0
            for arg in cfg_dict["call_para"]["args"]:
                if arg.get("dtype") is not None:
                    arg_dtype_num = len(arg["dtype"])
                    break

            if arg_dtype_num != 0:
                for i in range(arg_dtype_num):
                    for j in range(len(call_paras_list)):
                        tmp_cfg_dict = copy.deepcopy(cfg_dict)
                        tmp_cfg_dict["para"] = copy.deepcopy(para)
                        tmp_cfg_dict["call_para"]["args"] = copy.deepcopy(
                            call_paras_list[j])
                        if len(related_paras_list) != 0:
                            tmp_cfg_dict["related_para"] = copy.deepcopy(
                                related_paras_list[j])
                        for arg in tmp_cfg_dict["call_para"]["args"]:
                            if arg.get("dtype") is not None:
                                arg["dtype"] = copy.deepcopy(arg["dtype"][i])
                        cfg_expand_list.append(tmp_cfg_dict)
            # dtype does not exit in args, so do not take dtype into account
            else:
                for i in range(len(call_paras_list)):
                    tmp_cfg_dict = copy.deepcopy(cfg_dict)
                    tmp_cfg_dict["para"] = copy.deepcopy(para)
                    tmp_cfg_dict["call_para"]["args"] = copy.deepcopy(
                        call_paras_list[i])
                    if len(related_paras_list) != 0:
                        tmp_cfg_dict["related_para"] = copy.deepcopy(
                            related_paras_list[i])
                    cfg_expand_list.append(tmp_cfg_dict)
    return cfg_expand_list


def expand_cfg_by_all_options(cfg_dict: dict) -> list:
    paras_list, related_paras_list, call_paras_list = expand_cfg_by_para(cfg_dict)
    cfg_expand_list = expand_cfg_all(paras_list, related_paras_list, call_paras_list, cfg_dict)
    return cfg_expand_list


def delete_if_gen_fn_in_call_para(cfg_dict):
    for arg in cfg_dict["call_para"]["args"]:
        if "gen_fn" in arg.keys():
            arg.pop("gen_fn")

def delete_fn(cfg_dict):
    for arg in cfg_dict["call_para"]["args"]:
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
            value = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
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
        related_para_dict = cfg_dict["related_para"]
        call_para_args_list = cfg_dict["call_para"]["args"]

        for k in para_dict:
            construct_paras[k] = para_dict[k]
        for k in related_para_dict:
            construct_paras[k] = related_para_dict[k]

        for arg in call_para_args_list:
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
                assert(cfg_dict["call_para"]["seq_name"] != ""), "need a name the list of tensors"
        # tie all the function_paras in a list named seq_name
        if cfg_dict["call_para"]["seq_name"] != "":
            name = cfg_dict["call_para"]["seq_name"]
            new_list = []
            for j in tensor_list:
                new_list.append(j)
            for j in function_paras["kwargs"].values():
                new_list.append(j)
            function_paras["kwargs"][name] = new_list

        delete_if_gen_fn_in_call_para(cfg_dict)
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
            if func_name not in ['all', configs[cfg_name]['name']]:
                continue
            logger.info(f"generating input(s) for {cfg_name} ...")
            cfg_expand_list = expand_cfg_by_all_options(configs[cfg_name])
            cfg_counter += len(cfg_expand_list)
            gen_and_dump_data(inputs_dir_path, cfg_name, cfg_expand_list, cfg_save_dict)


        with open(os.path.join(inputs_dir_path, cfg_file_name), "wb") as f:
            pickle.dump(cfg_save_dict, f)

        logger.info(f"generate test cases number: {cfg_counter}")
        logger.info("generate input data done!")
