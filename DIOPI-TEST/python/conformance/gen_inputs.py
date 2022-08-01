import os
import sys
import copy
import pickle
from functools import partial

from .utils import logger
from .dtype import Dtype
from .testcase_parse import Genfunc, dict_elem_length


def _to_torch_dtype(val : Dtype):
    import torch
    if val == Dtype.float16:
        return torch.float16
    elif val == Dtype.float32:
        return torch.float32
    elif val == Dtype.float64:
        return torch.float64
    elif val == Dtype.int8:
        return torch.int8
    elif val == Dtype.int16:
        return torch.int16
    elif val == Dtype.int32:
        return torch.int32
    elif val == Dtype.int64:
        return torch.int64
    elif val == Dtype.uint8:
        return torch.uint8
    elif val == Dtype.bool:
        return torch.bool
    else:
        return torch.float32


def _to_torch_func(val : Genfunc):
    import torch
    if val == Genfunc.randn:
        return torch.randn
    elif val == Genfunc.rand:
        return torch.rand
    elif val == Genfunc.ones:
        return torch.ones
    elif val == Genfunc.zeros:
        return torch.zeros
    elif val == Genfunc.mask:
        return partial(torch.randint, high=2)
    elif val == Genfunc.empty:
        return torch.empty
    else:
        return torch.randn


def num_of_tensor(obj: dict):
    assert "shape" in obj.keys() or "value" in obj.keys()
    if "shape" in obj.keys():
        return len(obj["shape"])
    if "value" in obj.keys():
        return len(obj["value"])


def combination_dict_elem(obj: dict, key_idx, objs_out: list):
    r'''
    dict(a1=[1,2], a2=[11,22])
    ====>
    [dict(a1=1,a2=11),
    dict(a1=1, a2=22),
    dict(a1=2, a2=11),
    dict(a1=2, a2=22)]
    '''
    if key_idx >= len(obj):
        objs_out.append(obj)
    else:
        keys = list(obj.keys())
        for i in obj[keys[key_idx]]:
            new_obj = copy.deepcopy(obj)
            new_obj[keys[key_idx]] = i
            combination_dict_elem(new_obj, key_idx+1, objs_out)


def seq_dict_elem(obj: dict, objs_out: dict):
    r'''
    dict(a = [1,2], b = [11,22])
    --->
    [dict(a = 1, b = 11), dict(a=2, b=22)]
    '''
    length = dict_elem_length(obj)
    for i in range(length):
        new_obj = {}
        for obj_k, obj_v in obj.items():
            new_obj[obj_k] = copy.deepcopy(obj_v[i])
        objs_out.append(new_obj)


def seq_call_para_args(objs, objs_out):
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
    if len(objs) == 0 or objs is None:
        return
    # expand ins, requires_grad
    args = []
    for i_obj in range(len(objs)):
        for i_name in range(len(objs[i_obj]["ins"])):
            new_obj = copy.deepcopy(objs[i_obj])
            new_obj["ins"] = copy.deepcopy(objs[i_obj]["ins"][i_name])
            # new_obj["requires_grad"] = copy.deepcopy(
            #     objs[i_obj]["requires_grad"][i_name])
            args.append(new_obj)
    num = num_of_tensor(args[0])
    for i_tensor in range(num):
        new_args = copy.deepcopy(args)
        for i_args in range(len(args)):
            if "value" in args[i_args].keys():
                new_args[i_args]["value"] = copy.deepcopy(
                    args[i_args]["value"][i_tensor])
            elif "shape" in args[i_args].keys():
                new_args[i_args]["shape"] = copy.deepcopy(
                    args[i_args]["shape"][i_tensor])
        objs_out.append(new_args)


def combinate_args(case: dict):
    paras_out = []
    related_paras_out = []
    call_paras_args_out = []

    paras_out = []
    combination_dict_elem(case["para"], 0, paras_out)
    seq_dict_elem(case["related_para"], related_paras_out)
    seq_call_para_args(case["call_para"]["args"], call_paras_args_out)
    return paras_out, related_paras_out, call_paras_args_out


def obtain_dtype_num(args) -> int:
    for arg in args:
        if "dtype" in arg.keys():
            return len(arg["dtype"])
    return 0


# para: all, related and call: all ,type all
def generate_testcases(paras_comb, related_paras_comb, call_paras_args_comb, case_v) -> list:
    case_vs = []
    for i_para in range(len(paras_comb)):
        if len(call_paras_args_comb) != 0:
            dtype_num = obtain_dtype_num(case_v["call_para"]["args"])
            if dtype_num != 0:
                for i_dtype in range(dtype_num):
                    for i_call_para_args in range(len(call_paras_args_comb)):
                        new_case_v = copy.deepcopy(case_v)
                        new_case_v["para"] = copy.deepcopy(paras_comb[i_para])
                        new_case_v["call_para"]["args"] = copy.deepcopy(
                            call_paras_args_comb[i_call_para_args])
                        if len(related_paras_comb) != 0:
                            new_case_v["related_para"] = copy.deepcopy(
                                related_paras_comb[i_call_para_args])
                        for arg in new_case_v["call_para"]["args"]:
                            if 'dtype' in arg.keys():
                                arg["dtype"] = copy.deepcopy(arg["dtype"][i_dtype])
                        case_vs.append(new_case_v)
            # dtype does not exit in args, so do not take dtype into account
            else:
                for i_call_para_args in range(len(call_paras_args_comb)):
                    new_case_v = copy.deepcopy(case_v)
                    new_case_v["para"] = copy.deepcopy(paras_comb[i_para])
                    new_case_v["call_para"]["args"] = copy.deepcopy(
                        call_paras_args_comb[i_call_para_args])
                    if len(related_paras_comb) != 0:
                        new_case_v["related_para"] = copy.deepcopy(
                            related_paras_comb[i_call_para_args])
                    case_vs.append(new_case_v)
    return case_vs


def delete_fn(case_v):
    for arg in case_v["call_para"]["args"]:
        if "gen_fn" in arg.keys():
            arg.pop("gen_fn")
    return case_v


def gen_tensor(arg, case_v):
    import torch
    if "value" in arg.keys():
        value = torch.tensor(arg["value"], device=torch.device("cpu"))
    else:
        gen_f = _to_torch_func(arg["gen_fn"])
        try:
            if arg["shape"] is None:
                value = None
            else:
                dtype = _to_torch_dtype(arg["dtype"])
                # if arg["dtype"] == Dtype.float16:
                #     dtype = Dtype.float32
                value = gen_f(size=arg["shape"], dtype=dtype, device=torch.device("cpu"))
        except BaseException as e:
            logger.error(e, exc_info=True)
            logger.error(arg)
            sys.exit()
    if value is not None:
        value = value.numpy()
    return value


def gen_and_save_data(dir_path: str, case_k: str, case_vs: list, cfgs: dict):
    construct_paras = {}
    function_paras = {"kargs": [], "kwargs": {}}
    i = 0

    for case_v in case_vs:
        para = case_v["para"]
        related_para = case_v["related_para"]
        call_para_args = case_v["call_para"]["args"]

        for k, v in para.items():
            construct_paras[k] = v
        for k, v in related_para.items():
            construct_paras[k] = v

        for arg in call_para_args:
            name = arg["ins"]
            # length of gen_num_range must be 2, otherwise ignore gen_num_range
            if len(arg["gen_num_range"]) != 2:
                value = gen_tensor(arg, case_v)
                function_paras["kwargs"][name] = value
            else:
                tensors_num = random.randint(arg['gen_num_range'][0],
                                             arg['gen_num_range'][1])
                arg.setdefault("tensors_num", tensors_num)
                for j in range(tensors_num):
                    value = gen_tensor(arg, case_v)
                    function_paras["kargs"].append(value)
        # tie all the function_paras in a list named seq_name
        if case_v["call_para"]["seq_name"] != "":
            name = case_v["call_para"]["seq_name"]
            new_list = []
            for j in function_paras["kargs"]:
                new_list.append(j)
            for j in function_paras["kwargs"].values():
                new_list.append(j)
            function_paras["kargs"] = []
            function_paras["kwargs"][name] = new_list

        cfg = delete_fn(copy.deepcopy(case_v))

        function_paras["kwargs"].update(construct_paras)
        info = {"function_paras": function_paras, "cfg": cfg}
        file_name = f"{case_k}_{i}.pth"
        cfgs[file_name] = cfg
        with open(os.path.join(dir_path, file_name), "wb") as file:
            pickle.dump(info, file)

        function_paras["kargs"] = []
        function_paras['kwargs'] = {}
        i += 1


class GenData(object):
    def __init__(self, case_collection):
        self.case_collection = case_collection
        self.gen_num = 0

    def generate(self, tag=None):
        cases = self.case_collection.test_cases
        num = 0
        cfgs = {}
        dir_path = "data/inputs"
        for case_k, case_v in cases.items():
            logger.debug(f"generating {case_k} ...")
            paras_comb, related_paras_comb, call_paras_args_comb = combinate_args(case_v)
            case_vs = generate_testcases(paras_comb, related_paras_comb,
                                         call_paras_args_comb, case_v)
            num += len(case_vs)
            gen_and_save_data(dir_path, case_k, case_vs, cfgs)
            logger.debug("done")
        self.gen_num = num

        with open(os.path.join(dir_path, "cfgs.pth"), "wb") as cfg_file:
            pickle.dump(cfgs, cfg_file)

        logger.info(f"gen_num: {self.gen_num}")
        logger.info(f"generate input data done!")

