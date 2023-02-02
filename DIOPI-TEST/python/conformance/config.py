import copy
import pickle

from .utils import default_cfg_dict
from .dtype import Dtype


class Genfunc(object):
    randn = 0   # “standard normal” distribution
    rand = 1   # random samples from a uniform distribution over [0, 1).
    ones = 2
    zeros = 3
    mask = 4
    empty = 5
    randint = 6
    positive = 7
    sym_mat = 8
    default = 9

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as pkl_f:
            tensor = pickle.load(pkl_f)
        return tensor


def _must_be_the_type(cfg_path: str, cfg_dict: dict, required_type, cfg_keys: list) -> None:
    # def _must_be_the_type(cfg_dict: dict, cfg_keys: list, cfg_path: str, required_type) -> None:
    if isinstance(required_type, (list, tuple)):
        types_str = ""
        for i in required_type:
            types_str += i.__name__
            types_str += ' or '
        types_str = types_str[:-4]
    else:
        types_str = required_type.__name__

    err = f"key %s should be {types_str} in {cfg_path}"
    for key in cfg_keys:
        if key in cfg_dict.keys():
            assert isinstance(cfg_dict[key], required_type), err % key


def _must_be_list(cfg_dict: dict, cfg_keys: list, cfg_path: str):
    _must_be_the_type(cfg_dict, cfg_keys, cfg_path, list)


def _must_be_tuple(cfg_dict: dict, cfg_keys: list, cfg_path: str):
    _must_be_the_type(cfg_dict, cfg_keys, cfg_path, tuple)


def _must_not_iterable_in_list_or_tuple(domain: str, dict_obj: dict,
                                        required_type, keys: list):
    if isinstance(required_type, (list, tuple)):
        types_str = ""
        for i in required_type:
            types_str += i.__name__
            types_str += ' or '
        types_str = types_str[:-4]
    else:
        types_str = required_type.__name__

    err = f"key %s should be {types_str} in {domain}"
    for key in keys:
        assert isinstance(dict_obj[key], required_type), err % key


def _must_exist(domain, dict_obj, keys: list):
    err = f"key %s should set in {domain}"
    for key in keys:
        assert key in dict_obj.keys(), err % key


def dict_elem_length(dict_obj):
    if dict_obj == {} or dict_obj is None:
        return 0
    keys = list(dict_obj.keys())
    return len(dict_obj[keys[0]])


def check_dtype_not_nested_list_or_tuple(domain, dtype_obj):
    assert isinstance(dtype_obj, (list, tuple))
    for dt in dtype_obj:
        assert not isinstance(dt, (list, tuple)), \
            f"{domain} should not be nested list or tuple"


def check_configs_format(cfgs_dict: dict):
    for case_k, case_v in cfgs_dict.items():
        domain = f"diopi_configs.{case_k}"
        _must_be_the_type(domain, case_v, list, ["dtype", "pytorch"])
        # _must_be_list(case_v, ["dtype", "pytorch"], domain)

        _must_exist(domain, case_v, ['name'])
        _must_be_the_type(domain, case_v, list, ['name', 'arch'])

        if "tensor_para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['tensor_para'])
            if "dtype" in case_v.keys():
                _must_be_the_type(domain, case_v, list, ["dtype"])
                check_dtype_not_nested_list_or_tuple(f"{domain}.dtype",
                                                     case_v["dtype"])

            _must_exist(domain + ".tensor_para", case_v["tensor_para"], ["args"])
            _must_be_the_type(domain + ".tensor_para", case_v["tensor_para"],
                              (list, tuple), ['args'])
            domain_tmp = domain + ".tensor_para.args"
            dict_obj = case_v["tensor_para"]["args"]

            for arg in case_v["tensor_para"]['args']:
                _must_be_the_type(domain_tmp, arg, (list, tuple),
                                  [i for i in arg.keys() if i != "gen_fn"])
                if "gen_num_range" in arg.keys():
                    assert len(arg["gen_num_range"]) == 2, \
                        f"the length of {domain_tmp}.gen_num_range must be 2"
                for arg_k, arg_v in arg.items():
                    if arg_k == "dtype":
                        check_dtype_not_nested_list_or_tuple(
                            f"{domain_tmp}.{arg_k}.dtype", arg_v)

        if "para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['para'])
            dict_obj = case_v["para"]
            _must_be_the_type(domain + ".para", dict_obj, (list, tuple),
                              [i for i in dict_obj.keys() if i != "gen_fn"])

        # checking the length associated with para
        length = 0
        if "para" in case_v.keys():
            domain_tmp = domain + ".para"
            length = dict_elem_length(case_v["para"])
            for para_k, para_v in \
                    case_v["para"].items():
                assert length == len(para_v), \
                    "the length of " + domain_tmp + "." + para_k + \
                    "should equal " + str(length)
            if "tensor_para" in case_v.keys():
                if "args" in case_v["tensor_para"]:
                    args: list = case_v["tensor_para"]["args"]
                    domain_tmp0 = domain_tmp + "tensor_para.args"
                    for arg in args:
                        if "shape" in arg.keys():
                            assert length == len(arg["shape"]), \
                                f"the length of {domain_tmp0}.shape" + \
                                " should equal " + str(length)
                        if "value" in arg.keys():
                            assert length == len(arg["value"]), \
                                f"the length of {domain_tmp0}.value" + \
                                " should equal " + str(length)


# expand test config according to key
def expand_cfg_by_name(cfgs_dict: dict, key: str) -> dict:
    '''
    test: {
        ...
        key: ["value1", "value2"],
        ...
    }
    ====>
    test::value1: {
        ...
        key: "value1",
        ...
    }
    test::value2: {
        ...
        key: "value2",
        ...
    }
    '''
    expand_cfg_dict = {}
    for cfg_name in cfgs_dict:
        expand_values = cfgs_dict[cfg_name].get(key)
        expand_values = expand_values if isinstance(expand_values, (list, tuple)) else \
            [expand_values]
        # delete the key named name in diopi_configs.
        for value in expand_values:
            expand_cfg_value = copy.deepcopy(cfgs_dict[cfg_name])
            expand_cfg_value[key] = value

            if not isinstance(value, str):
                value = value.__str__
            # add new key named {name}_{value} into test_case
            expand_cfg_dict[f"{cfg_name}::{value}"] = expand_cfg_value
    return expand_cfg_dict


def append_default_cfg_options(cfgs_dict):
    default_option = default_cfg_dict["default_option"]
    for key in cfgs_dict:
        cfg_value = cfgs_dict[key]
        for key in default_option:
            if key not in cfg_value:
                cfg_value[key] = default_option[key]


def add_name_for_tensor_config(cfgs_dict):
    for key in cfgs_dict:
        if "name" not in cfgs_dict[key]:
            cfgs_dict[key]["name"] = [key]


def check_and_expand_in_args(domain, args: dict, key):
    length = 1
    len_dtype_eq_1 = []
    len_dtype_not_eq_1 = []
    for arg in args:
        if key in arg.keys():
            if len(arg[key]) == 1:
                len_dtype_eq_1.append(arg)
            else:
                len_dtype_not_eq_1.append(arg)

    is_1st = True
    for arg in len_dtype_not_eq_1:
        if key in arg.keys():
            if is_1st:
                length = len(arg[key])
            else:
                if length != len(arg[key]):
                    assert False, f"{domain}.{key} length is not matched"

    for arg in len_dtype_eq_1:
        if key in arg.keys():
            if length != 1:
                arg[key] = arg[key] * length


def delete_key_if_exist(case_v, key):
    if key in case_v.keys():
        case_v.pop(key)
    if "tensor_para" in case_v.keys():
        if key in case_v["tensor_para"].keys():
            case_v["tensor_para"].pop(key)


def check_and_set(case_v, key, default_v):
    for item in case_v["tensor_para"]["args"]:
        if key not in item.keys():
            if key not in case_v["tensor_para"].keys():
                if key not in case_v.keys():
                    item[key] = default_v
                else:
                    item[key] = case_v[key]
            else:
                item[key] = case_v["tensor_para"][key]


def format_cfg(cases):
    for case_k, case_v in cases.items():
        # set [] for defalut para, tensor_para, para, pytorch
        # if "pytorch" not in case_v.keys():
        #     case_v["pytorch"] = pt.default
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
        if "skip_if" not in case_v.keys():
            case_v["skip_if"] = ""
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
            if "gen_num_range" not in item.keys():
                item["gen_num_range"] = []
        # gen_fn and dtype maybe set in global zone,
        # we don't recommend the set key in global zone.
        check_and_set(case_v, "dtype", Dtype.default)
        check_and_set(case_v, "gen_fn", Genfunc.default)

        domain = f"{case_k}.tensor_para.args"
        check_and_expand_in_args(domain, case_v["tensor_para"]["args"], "dtype")
        check_and_expand_in_args(domain, case_v["tensor_para"]["args"], "shape")
        # delete the keys dtype and gen_fn which are not in the right positions
        delete_key_if_exist(case_v, "dtype")
        delete_key_if_exist(case_v, "gen_fn")


class Config(object):
    r"""
    Process config file
    """

    @staticmethod
    def process_configs(cfgs_dict: dict):
        check_configs_format(cfgs_dict)
        cfgs_dict = expand_cfg_by_name(cfgs_dict, 'name')
        append_default_cfg_options(cfgs_dict)
        format_cfg(cfgs_dict)
        return cfgs_dict
