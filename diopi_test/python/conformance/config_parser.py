import os
import sys
import copy
import pickle
import itertools
import numpy as np
from enum import Enum


def diopi_config_parse(case_item_path="./cache/diopi_case_items.cfg"):
    """
    todo: config file defined by user
    """
    sys.path.append("../python/configs")
    from diopi_configs import diopi_configs

    ci = ConfigParser(ofile=case_item_path)
    ci.parser(diopi_configs)
    ci.save()


class ConfigParser(object):
    def __init__(self, ofile="test_config.cfg") -> None:
        self._ofile = ofile
        self._items = {}

    def __str__(self):
        return str(self._items)

    def parser(self, config, fname="all_ops"):
        if isinstance(config, dict):
            self._config_dict_parse(config, fname)
        elif os.path.isfile(config):
            raise NotImplementedError("Will sopport config file.")
        elif os.path.isdir(config):
            raise NotImplementedError("Will sopport config files under given path.")
        else:
            raise Exception(f"{__file__}: Parameter {config} passed with wrong type.")

    def _config_dict_parse(self, config, fname):
        for key, value in config.items():
            if fname != "all_ops" and fname not in value["name"]:
                continue
            cfg_item = ConfigItem(key, value)
            cfg_item.generate_items()
            case_items = cfg_item.get_case_items()
            self._items.update(case_items)

    def get_config_cases(self):
        return self._items

    def reset(self):
        self._items = {}

    def save(self, path=""):
        path = path if path != "" else self._ofile
        with open(path, "wb") as f:
            pickle.dump(self._items, f)


# *********************************************************************************
# internal helper function
# *********************************************************************************
def _assert_exist(cfg_name, cfg_dict, keys):
    err = f"key %s not in {cfg_name}"
    for key in keys:
        assert key in cfg_dict.keys(), err % key


def _assert_type(cfg_name, cfg_dict, require_type, item_keys):
    if isinstance(require_type, (list, tuple)):
        types_str = ""
        for t in require_type:
            types_str += t.__name__
            types_str += " or "
        types_str = types_str[:-4]
    else:
        types_str = require_type.__name__

    err = f"key %s: should be {types_str} in {cfg_name} items"
    for k in item_keys:
        if k in cfg_dict.keys():
            assert isinstance(cfg_dict[k], require_type), err % k


def _assert_unnested_type(cfg_name, obj):
    assert isinstance(obj, (list, tuple))
    for o in obj:
        assert not isinstance(
            o, (list, tuple)
        ), f"{cfg_name} should not be nested list or tuple"


def _check_and_expand_in_args(domain, args: dict, key):
    length = 1
    len_value_eq_1 = []
    len_value_not_eq_1 = []
    for arg in args:
        if key in arg.keys():
            if len(arg[key]) == 1:
                len_value_eq_1.append(arg)
            else:
                len_value_not_eq_1.append(arg)

    is_1st = True
    for arg in len_value_not_eq_1:
        if key in arg.keys():
            if is_1st:
                length = len(arg[key])
            else:
                if length != len(arg[key]):
                    assert False, f"{domain}.{key} length is not matched"

    for arg in len_value_eq_1:
        if key in arg.keys():
            if length != 1:
                arg[key] = arg[key] * length


def _tensor_para_default(case_v, key, default_v):
    for item in case_v["tensor_para"]["args"]:
        if key not in item.keys():
            if key not in case_v["tensor_para"].keys():
                if key not in case_v.keys():
                    item[key] = default_v
                else:
                    item[key] = case_v[key]
            else:
                item[key] = case_v["tensor_para"][key]


# expand elements
def _dict_elem_length(dict_obj):
    if dict_obj == {} or dict_obj is None:
        return 0
    keys = list(dict_obj.keys())
    return len(dict_obj[keys[0]])


def _expand_para(para_dict: dict, paras_list: list):
    r"""
    dict(a = [1,2], b = [11,22])
    --->
    [dict(a = 1, b = 11), dict(a=2, b=22)]
    """
    length = _dict_elem_length(para_dict)
    for i in range(length):
        tmp_para_dict = {}
        for k, v in para_dict.items():
            tmp_para_dict[k] = copy.deepcopy(v[i])
        paras_list.append(tmp_para_dict)


def _expand_tensor_para(args_list, tensor_paras_list):
    r"""
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
    """
    if len(args_list) == 0 or args_list is None:
        return
    # expand ins, requires_grad
    tmp_args_list = []
    for arg in args_list:
        for i in range(len(arg["ins"])):
            tmp_arg = copy.deepcopy(arg)
            tmp_arg["ins"] = copy.deepcopy(arg["ins"][i])
            tmp_args_list.append(tmp_arg)

    args0_dict = tmp_args_list[0]
    assert "shape" in args0_dict or "value" in args0_dict
    num = (
        len(args0_dict["shape"]) if "shape" in args0_dict else len(args0_dict["value"])
    )

    for j in range(num):
        args_ins_expand_list = copy.deepcopy(tmp_args_list)
        for i in range(len(tmp_args_list)):
            stride_name = str(tmp_args_list[i]["ins"]) + "stride"
            if "value" in tmp_args_list[i].keys():
                args_ins_expand_list[i]["value"] = copy.deepcopy(
                    tmp_args_list[i]["value"][j]
                )
            elif "shape" in tmp_args_list[i].keys():
                args_ins_expand_list[i]["shape"] = copy.deepcopy(
                    tmp_args_list[i]["shape"][j]
                )
            if "stride" in tmp_args_list[i].keys():
                if j >= len(args0_dict["stride"]):
                    del args_ins_expand_list[i]["stride"]
                    continue
                elif tmp_args_list[i]["stride"][j] is None:
                    del args_ins_expand_list[i]["stride"]
                    continue
                args_ins_expand_list[0][stride_name] = copy.deepcopy(
                    tmp_args_list[i]["stride"][j]
                )
                # 判断stride和shape是否符合标准，不符合报错
                tmp_stride = args_ins_expand_list[0][stride_name]
                tmp_shape = args_ins_expand_list[i]["shape"]
                assert len(tmp_stride) == len(
                    tmp_shape
                ), "stride and shape must have the same dim"
                stride_dic = []
                for index, s, st in zip(
                    range(len(list(tmp_shape))), list(tmp_shape), tmp_stride
                ):
                    stride_dic.append((s, st))
                sorted_stride = sorted(stride_dic, key=lambda x: x[1])
                for index in range(len(sorted_stride) - 1):
                    assert (sorted_stride[index][0] - 1) * sorted_stride[index][
                        1
                    ] < sorted_stride[index + 1][
                        1
                    ], "wrong stride for shape (might have memory overlap)"
        tensor_paras_list.append(args_ins_expand_list)


def _expand_config_with_para(config_item: dict):
    paras_list = []
    tensor_paras_list = []

    _expand_para(config_item["para"], paras_list)
    _expand_tensor_para(config_item["tensor_para"]["args"], tensor_paras_list)
    return paras_list, tensor_paras_list


def _expand_config_all(conif_item: dict, paras_list, tensor_paras_list):
    cfg_expand_list = []
    filter_dtype_list = []
    if len(tensor_paras_list) != 0:
        arg_dtype_num = 0
        for arg in conif_item["tensor_para"]["args"]:
            if arg.get("dtype") is not None:
                arg_dtype_num = len(arg["dtype"])
                break

        if arg_dtype_num != 0:
            for i in range(arg_dtype_num):
                for j in range(len(tensor_paras_list)):
                    filter_dtype = False
                    tmp_cfg_dict = copy.deepcopy(conif_item)
                    tmp_cfg_dict["tensor_para"]["args"] = copy.deepcopy(
                        tensor_paras_list[j]
                    )
                    if len(paras_list) != 0:
                        tmp_cfg_dict["para"] = copy.deepcopy(paras_list[j])
                    for arg in tmp_cfg_dict["tensor_para"]["args"]:
                        if arg.get("dtype") is not None:
                            entry_dtype = arg["dtype"][i]
                            if entry_dtype in filter_dtype_list:
                                filter_dtype = True
                                break
                            else:
                                arg["dtype"] = copy.deepcopy(entry_dtype)
                    if not filter_dtype:
                        cfg_expand_list.append(tmp_cfg_dict)
        # dtype does not exit in args, so do not take dtype into account
        else:
            for i in range(len(tensor_paras_list)):
                tmp_cfg_dict = copy.deepcopy(conif_item)
                tmp_cfg_dict["para"] = copy.deepcopy(paras_list[i])
                tmp_cfg_dict["tensor_para"]["args"] = copy.deepcopy(
                    tensor_paras_list[i]
                )
                cfg_expand_list.append(tmp_cfg_dict)
    elif len(paras_list) != 0:
        for i in range(len(paras_list)):
            tmp_cfg_dict = copy.deepcopy(conif_item)
            tmp_cfg_dict["para"] = copy.deepcopy(paras_list[i])
            cfg_expand_list.append(tmp_cfg_dict)

    return cfg_expand_list


def _expand_config_with_all(config_item: dict):
    paras_list, tensor_para_list = _expand_config_with_para(config_item)
    # print(colored(f"{len(paras_list)} == {len(tensor_para_list)}", 'yellow'))
    config_case_items = _expand_config_all(config_item, paras_list, tensor_para_list)

    # print(colored(f"{len(config_case_items)}", 'red'))
    return config_case_items


class ConfigItem(object):
    def __init__(self, item_name="batch_norm", item_config: dict = {}) -> None:
        self._name = item_name
        self._orig = item_config

        # cache result
        self._config_items = {}  # {'batch'}
        self._case_items = {}

    def __str__(self) -> str:
        return str(self._case_items)

    def _check_format(self):
        _assert_type(self._name, self._orig, list, ["dtype", "pytorch"])
        _assert_exist(self._name, self._orig, ["name"])
        _assert_type(self._name, self._orig, list, ["name", "arch"])
        # tensor_para
        if "tensor_para" in self._orig.keys():
            _assert_type(self._name, self._orig, dict, ["tensor_para"])
            if "dtype" in self._orig.keys():
                _assert_type(self._name, self._orig, list, ["dtype"])
                _assert_unnested_type(self._name, self._orig["dtype"])
            _assert_exist(
                self._name + ".tensor_para", self._orig["tensor_para"], ["args"]
            )
            _assert_type(
                self._name + ".tensor_para",
                self._orig["tensor_para"],
                (list, tuple),
                ["args"],
            )

            # check args: []
            args_name = self._name + ".tensor_para.args"
            for arg in self._orig["tensor_para"]["args"]:
                _assert_type(
                    args_name,
                    arg,
                    (list, tuple),
                    [k for k in arg.keys() if k not in ["gen_fn", "gen_policy"]],
                )
                # should design gen policy: map
                for k, v in arg.items():
                    if k == "dtype":
                        _assert_unnested_type(args_name + f"{k}.dtype", v)

        if "para" in self._orig.keys():
            _assert_type(self._name, self._orig, dict, ["para"])
            para_obj = self._orig["para"]
            # is there gen_fn in para dict ？
            _assert_type(
                self._name + ".para",
                para_obj,
                (list, tuple),
                [k for k in para_obj.keys()],
            )

        length = 0
        if "para" in self._orig.keys():
            para_obj = self._orig["para"]
            for k, v in para_obj.items():
                if length == 0:
                    length = len(v)
                else:
                    assert (
                        len(v) == length
                    ), f"{self._name}.para.{k}: length not matched."

            if "tensor_para" in self._orig.keys():
                # shape, value
                args = self._orig["tensor_para"]["args"]
                for i, arg in enumerate(args):
                    if "shape" in arg.keys():
                        assert (
                            len(arg["shape"]) == length
                        ), f"{self._name}.tensor_para.args[{i}].shape: length not matched."
                    if "value" in arg.keys():
                        assert (
                            len(arg["value"]) == length
                        ), f"{self._name}.tensor_para.args[{i}].value: length not matched."

    def _expand_by_name(self, key="name"):
        expand_names = self._orig[key]
        expand_names = (
            expand_names if isinstance(expand_names, (list, tuple)) else [expand_names]
        )
        for en in expand_names:
            cfg_item = copy.deepcopy(self._orig)
            cfg_item[key] = en
            self._config_items[f"{self._name}::{str(en)}"] = cfg_item

    def _expand_config_items(self):
        for key in self._config_items:
            # func_name = self._config_items[key]['name']
            config_items_list = _expand_config_with_all(self._config_items[key])
            # print(colored(f'config_item_list = {len(config_items_list)}', 'blue'))
            # case_sn = 0
            for sn, cil in enumerate(config_items_list):
                self._case_items[key + f"_{sn}.pth"] = CaseItem(cil).get_item()
                # case_sn += 1

    def _config_format(self):
        for case_k, case_v in self._config_items.items():
            # set [] for defalut para, tensor_para, para
            if "tensor_para" not in case_v.keys():
                case_v["tensor_para"] = {}
            if "args" not in case_v["tensor_para"].keys():
                case_v["tensor_para"]["args"] = []
            if "seq_name" not in case_v["tensor_para"].keys():
                case_v["tensor_para"]["seq_name"] = ""

            if "requires_backward" not in case_v:
                case_v["requires_backward"] = []
            if "para" not in case_v.keys():
                case_v["para"] = {}
            if "tag" not in case_v.keys():
                case_v["tag"] = []
            if "saved_args" not in case_v.keys():
                case_v["saved_args"] = {}
            if "interface" not in case_v.keys():
                case_v["interface"] = []
            # set default value of ins with ["input"] and
            # requires_grad with False
            for item in case_v["tensor_para"]["args"]:
                if "ins" not in item.keys():
                    item["ins"] = ["input"]
                if "requires_grad" not in item.keys():
                    item["requires_grad"] = [False]
                    # item["requires_grad"] = False
                if "gen_num_range" not in item.keys():
                    item["gen_num_range"] = []
            # gen_fn and dtype maybe set in global zone,
            # we don't recommend the set key in global zone.
            _tensor_para_default(
                case_v,
                "dtype",
                [np.float16, np.float32, np.float64, np.int32, np.int64],
            )
            _tensor_para_default(case_v, "gen_fn", "Genfunc.randn")
            _tensor_para_default(case_v, "gen_policy", "default")

            case_sig = f"{case_k}.tensor_para.args"
            _check_and_expand_in_args(case_sig, case_v["tensor_para"]["args"], "dtype")
            _check_and_expand_in_args(case_sig, case_v["tensor_para"]["args"], "shape")

            if "dtype" in case_v:
                case_v.pop("dtype")
            if "gen_fn" in case_v:
                case_v.pop("gen_fn")
            if "gen_policy" in case_v:
                case_v.pop("gen_policy")

    def generate_items(self):
        # check format and expand by 'name'
        self._check_format()
        self._expand_by_name()
        self._config_format()
        # expand other keys in config_items
        self._expand_config_items()

    def get_case_items(self):
        return self._case_items


# a case item
class CaseItem(object):
    def __init__(self, item: dict = {}) -> None:
        self._item = {
            "atol": 1e-5,
            "rtol": 1e-5,
            "atol_half": 1e-2,
            "rtol_half": 5e-2,
            "mismatch_ratio_threshold": 1e-3,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "gen_policy": "dafault",
        }
        for key, val in item.items():
            self._item[key] = val

    def __str__(self) -> str:
        # print(f"{__file__}:{self.__class__}:{self.__module__}")
        return str(self._item)

    def get_item(self) -> dict:
        return self._item

    def set_attr_val(self, key, val) -> None:
        if key not in self._item.keys():
            raise KeyError("Config {key} is not Item keys.")
        self._item[key] = val

    def add_attr_val(self, key, val) -> None:
        if key in self._item.keys():
            raise KeyError("Config {key} setted in Item keys.")
        self._item[key] = val


if __name__ == "__main__":
    # only for module test
    diopi_config_parse()
    # import sys
    # sys.path.append('../python/configs')
    # from diopi_configs import diopi_configs
    # print("cfgs:", diopi_configs.keys())

    # cp = ConfigParser()
    # cp.parser(diopi_configs)
    # print(cp._items['batch_norm::batch_norm_0.pth'])
    # cp.save()

    # from diopi_configs_v2 import diopi_configs
    # print(type(diopi_configs))

    # cfg_item = diopi_configs['batch_norm']
    # ci = ConfigItem('batch_norm', cfg_item)
    # # print(ci._orig)
    # ci.generate_items()
    # print(ci._case_items)
    # # print(ci._case_items['batch_norm::batch_norm_0.pth'])
    # # for i in range(11):
    # #     print(ci._case_items[f'batch_norm::batch_norm_{1}.pth']['tensor_para']['args'][0]['dtype'])
    # cfg_item = diopi_configs['ctc_loss']
    # ci = ConfigItem('ctc_loss', cfg_item)
    # # print(ci._orig)
    # ci.generate_items()
    # print(ci._case_items)

    # cfg_item = diopi_configs['pointwise_op']
    # print(cfg_item)
    # ci = ConfigItem('pointwise_op', cfg_item)
    # # print(ci._orig)
    # ci.generate_items()
    # print(ci._case_items)
    # cp = ConfigParser()
    # cp.parser(diopi_configs)
    # print(cp._items['batch_norm::batch_norm_0.pth'])
    # cp.save()

    # cp.reset()
    # cp.parser("test_config.cfg")
