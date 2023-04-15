import copy
from .config import _must_be_the_type, _must_exist, expand_cfg_by_name


class Skip:
    def __init__(self, value):
        self.value = value


def _must_be_the_list_or_tuple_of_type(cfg_path: str, cfg_dict: dict, required_type, cfg_keys: list) -> None:
    if isinstance(required_type, (list, tuple)):
        types_str = ""
        for i in required_type:
            types_str += i.__name__
            types_str += ' or '
        types_str = types_str[:-4]
    else:
        types_str = required_type.__name__

    err = f"key %s should be the list or tuple of {types_str} in {cfg_path}"
    for key in cfg_keys:
        if key in cfg_dict.keys():
            assert isinstance(cfg_dict[key], (list, tuple)), err % key
            for v in cfg_dict[key]:
                assert isinstance(v, required_type), err % key


def check_configs_format(cfgs_dict: dict):
    for case_k, case_v in cfgs_dict.items():
        domain = f"device_configs.{case_k}"
        _must_be_the_type(domain, case_v, list, ["dtype"])
        if "dtype" in case_v.keys():
            _must_be_the_list_or_tuple_of_type(domain, case_v, Skip, ["dtype"])

        _must_exist(domain, case_v, ['name'])
        _must_be_the_type(domain, case_v, list, ['name'])

        if "tensor_para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['tensor_para'])
            _must_exist(domain + ".tensor_para", case_v["tensor_para"], ["args"])
            _must_be_the_type(domain + ".tensor_para", case_v["tensor_para"],
                              (list, tuple), ['args'])
            domain_tmp = domain + ".tensor_para.args"
            for arg in case_v["tensor_para"]['args']:
                _must_exist(domain_tmp, arg, ["ins"])
                _must_be_the_list_or_tuple_of_type(domain_tmp, arg, Skip, ['shape', 'value', 'dtype'])

        if "para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['para'])
            dict_obj = case_v["para"]
            _must_be_the_list_or_tuple_of_type(domain + ".para", dict_obj, Skip,
                                               [i for i in dict_obj.keys()])


def expand_tensor_paras_args_by_ins(cfgs_dict):
    '''
    [
        {
            "ins": ['x1', 'x2'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
    ]
    ====>
    {
        'x1':{
            "ins": ['x1'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
        'x2':{
            "ins": ['x2'],
            "shape": [(2, 3, 16), (4, 32, 7, 7)],
        },
    }
    '''
    for cfg_name in cfgs_dict:
        tensor_para_args = cfgs_dict[cfg_name]["tensor_para"]["args"]
        tmp_tensor_para_args = {}
        for arg in tensor_para_args:
            assert isinstance(arg["ins"], (list, tuple))
            for in_name in arg["ins"]:
                tmp_tensor_para_args[in_name] = copy.deepcopy(arg)
                tmp_tensor_para_args[in_name]["ins"] = [in_name]
        cfgs_dict[cfg_name]["tensor_para"]["args"] = tmp_tensor_para_args


def format_cfg(cases):
    for case_k, case_v in cases.items():
        # set [] for defalut para, tensor_para, para
        if "tensor_para" not in case_v.keys():
            case_v["tensor_para"] = {}
        if "args" not in case_v["tensor_para"].keys():
            case_v["tensor_para"]["args"] = []
        if "para" not in case_v.keys():
            case_v["para"] = {}


def extract_value_from_skip(cfgs_dict):
    for case_k, case_v in cfgs_dict.items():
        if "dtype" in case_v.keys():
            case_v["dtype"] = [x.value for x in case_v["dtype"]]
        for para_k, para_v in case_v["para"].items():
            case_v["para"][para_k] = [x.value for x in para_v]
        for arg_k, arg_v in case_v["tensor_para"]["args"].items():
            if "shape" in arg_v:
                arg_v["shape"] = [x.value for x in arg_v["shape"]]
            if "value" in arg_v:
                arg_v["value"] = [x.value for x in arg_v["value"]]
            if "dtype" in arg_v:
                arg_v["dtype"] = [x.value for x in arg_v["dtype"]]


class DeviceConfig(object):
    r"""
    Process device config file
    """

    @staticmethod
    def process_configs(cfgs_dict: dict):
        check_configs_format(cfgs_dict)
        cfgs_dict = expand_cfg_by_name(cfgs_dict, 'name')
        format_cfg(cfgs_dict)
        expand_tensor_paras_args_by_ins(cfgs_dict)
        extract_value_from_skip(cfgs_dict)
        return cfgs_dict
