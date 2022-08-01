import copy
import pickle

from .utils import default_vals
from .dtype import Dtype


class Genfunc(object):
    randn   = 0
    rand    = 1
    ones    = 2
    zeros   = 3
    mask    = 4
    empty   = 5
    default = 6

    # @staticmethod
    # def target(size, dtype, device, requires_grad=False):
    #     return torch.ones(size=size, dtype=dtype, device=device).masked_fill_(
    #         Genfunc.mask(size=size, dtype=torch.uint8, device=device), -1)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as pkl_f:
            tensor = pickle.load(pkl_f)
        return tensor


def _must_be_the_type(domain: str, dict_obj: dict, required_type, keys: list):
    '''
    domain:
        "name":dict_obj
    '''
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
        if key in dict_obj.keys():
            assert isinstance(dict_obj[key], required_type), err % key


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


def _must_not_exist(domain, dict_obj, keys: list):
    err = f"key %s should not set in {domain}"
    for key in keys:
        assert key not in dict_obj.keys(), err % key


def dict_elem_length(dict_obj):
    if dict_obj == {} or dict_obj is None:
        return 0
    keys = list(dict_obj.keys())
    return len(dict_obj[keys[0]])


def check_cases(tag: str, cases: dict):
    def check_dtype_not_nested_list_or_tuple(domain, dtype_obj):
        assert isinstance(dtype_obj, (list, tuple))
        for dt in dtype_obj:
            assert not isinstance(dt, (list, tuple)), \
                f"{domain} should not be nested list or tuple"
    for case_k, case_v in cases.items():
        domain = f"{tag}_config.{case_k}"
        _must_be_the_type(domain, case_v, list, ["dtype", "pytorch"])

        _must_exist(domain, case_v, ['name'])
        _must_be_the_type(domain, case_v, list, ['name', 'arch'])

        if "para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['para'])
            dict_obj = case_v["para"]
            _must_be_the_type(domain+".para", dict_obj, (list, tuple),
                              [i for i in dict_obj.keys() if i != "gen_fn"])

        if "call_para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['call_para'])
            if "dtype" in case_v.keys():
                _must_be_the_type(domain, case_v, list, ["dtype"])
                check_dtype_not_nested_list_or_tuple(f"{domain}.dtype",
                                                     case_v["dtype"])

            _must_exist(domain + ".call_para", case_v["call_para"], ["args"])
            _must_be_the_type(domain + ".call_para", case_v["call_para"],
                              (list, tuple), ['args'])
            domain_tmp = domain + ".call_para.args"
            dict_obj = case_v["call_para"]["args"]

            for arg in case_v["call_para"]['args']:
                _must_be_the_type(domain_tmp, arg, (list, tuple),
                                  [i for i in arg.keys() if i != "gen_fn"])
                if "gen_num_range" in arg.keys():
                    assert len(arg["gen_num_range"]) == 2, \
                        f"the length of {domain_tmp}.gen_num_range must be 2"
                for arg_k, arg_v in arg.items():
                    if arg_k == "dtype":
                        check_dtype_not_nested_list_or_tuple(
                            f"{domain_tmp}.{arg_k}.dtype", arg_v)

        if "related_para" in case_v.keys():
            _must_be_the_type(domain, case_v, dict, ['related_para'])
            dict_obj = case_v["related_para"]
            _must_be_the_type(domain+".related_para", dict_obj, (list, tuple),
                              [i for i in dict_obj.keys() if i != "gen_fn"])

        # checking the length associated with related_para
        length = 0
        if "related_para" in case_v.keys():
            domain_tmp = domain+".related_para"
            length = dict_elem_length(case_v["related_para"])
            for related_para_k, related_para_v in \
                    case_v["related_para"].items():
                assert length == len(related_para_v), \
                    "the length of " + domain_tmp + "." + related_para_k + \
                    "should equal " + str(length)
            if "call_para" in case_v.keys():
                if "args" in case_v["call_para"]:
                    args: list = case_v["call_para"]["args"]
                    domain_tmp0 = domain_tmp + "call_para.args"
                    for arg in args:
                        if "shape" in arg.keys():
                            assert length == len(arg["shape"]), \
                                f"the length of {domain_tmp0}.shape" + \
                                " should equal " + str(length)
                        if "value" in arg.keys():
                            assert length == len(arg["value"]), \
                                f"the length of {domain_tmp0}.value" + \
                                " should equal " + str(length)


# expand test cases according to expd
def expand_cases(test_cases: dict, expd: str) -> dict:
    '''
    test1: {
        name: "op1",
        expd: ["expd1", "expd2"]
    }
    ====>
    test1::expd1: {
        name: "op1",
        expd: "expd1"
     }
     test1::expd2:{
        name: "op1",
        expd: "expd2"
     }
    '''
    new_cases = {}
    for name, attr in test_cases.items():
        expd_val = attr.get(expd)
        expd_val = expd_val if isinstance(expd_val, (list, tuple)) else \
            [expd_val]
        # delete the key named name in configs.
        for item in expd_val:
            new_attr = copy.deepcopy(attr)
            new_attr[expd] = item

            if not isinstance(item, str):
                item = item.__str__
            # add new key named {name}_{item} into test_case
            new_cases[f"{name}::{item}"] = new_attr
    return new_cases


# process property "base"
def derived_cases(cases: dict, base: str, is_self=True,
                  other_cases: dict = {}, pop_keys: list = []) -> dict:
    r'''
        Args:
            cases: the dict values that will be check
            base_name: the replacement tag
            is_self: search in cases if true else extend_cases
            extend_cases: if not is_self, find base_name in this.
            pop_keys: the keys in other_cases that should be popped
    '''

    if is_self:
        case_bank = cases
        assert pop_keys == []
        assert other_cases == {}
    else:
        assert other_cases != {} and other_cases is not None, \
            f"when is_self is {is_self}, extend_cases should not be None"
        case_bank = other_cases

    for name, attr in cases.items():
        base_name = attr.get(base, None)
        if base_name is None:
            continue
        elif base_name not in case_bank.keys():
            assert False, f"{base_name} is not found in case bank"
        else:
            new_attr = copy.deepcopy(case_bank[base_name])
            for key in pop_keys:
                new_attr.pop(key)
            new_attr.update(attr)

            new_attr.pop(base)
            cases[name] = new_attr

    return cases


def add_default_val(cases):
    defaults = default_vals["test_case_paras"]
    for case in cases.items():
        case_v = case[1]
        for key in defaults:
            if key not in case_v.keys():
                case_v[key] = defaults[key]


def add_name_for_tensor_config(cases):
    for case_k, case_v in cases.items():
        if "name" not in case_v.keys():
            case_v["name"] = [case_k]


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
    if "call_para" in case_v.keys():
        if key in case_v["call_para"].keys():
            case_v["call_para"].pop(key)


def check_and_set(case_v, key, default_v):
    for item in case_v["call_para"]["args"]:
        if "value" not in item.keys():
            if key not in item.keys():
                if key not in case_v["call_para"].keys():
                    if key not in case_v.keys():
                        item[key] = default_v
                    else:
                        item[key] = case_v[key]
                else:
                    item[key] = case_v["call_para"][key]


def format_cfg(cases):
    for case_k, case_v in cases.items():
        # set [] for defalut para, call_para, related_para, pytorch
        # if "pytorch" not in case_v.keys():
        #     case_v["pytorch"] = pt.default
        if "call_para" not in case_v.keys():
            case_v["call_para"] = {}
        if "args" not in case_v["call_para"].keys():
            case_v["call_para"]["args"] = []
        if "seq_name" not in case_v["call_para"].keys():
            case_v["call_para"]["seq_name"] = ""

        if "requires_backward" not in case_v:
            case_v["requires_backward"] = []
        if "para" not in case_v.keys():
            case_v["para"] = {}
        if "related_para" not in case_v.keys():
            case_v["related_para"] = {}
        if "skip_if" not in case_v.keys():
            case_v["skip_if"] = ""
        if "saved_args" not in case_v.keys():
            case_v["saved_args"] = []
        # set default value of ins with ["input"] and
        # requires_grad with False
        for item in case_v["call_para"]["args"]:
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

        domain = f"{case_k}.call_para.args"
        check_and_expand_in_args(domain, case_v["call_para"]["args"], "dtype")
        check_and_expand_in_args(domain, case_v["call_para"]["args"], "shape")
        # delete the keys dtype and gen_fn which are not in the right positions
        delete_key_if_exist(case_v, "dtype")
        delete_key_if_exist(case_v, "gen_fn")


def normalize_cases(cases) -> tuple:
    if cases is None:
        case_num = 0
    else:
        # expand it, after that the case will only belong to one func
        cases = expand_cases(cases, 'name')
        case_num = len(cases)
        # add default_val for all the cases
        add_default_val(cases)
        format_cfg(cases)
        return cases, case_num


class CaseCollection(object):
    def __init__(self, configs: dict):
        self.test_cases: dict = configs

        self.case_num = 0
        self.total_cases = {}

        # process all the cases
        self.process_cases()

    # preprocess cases and check the keys
    def preprocess_and_check(self):
        self.test_cases = derived_cases(self.test_cases, "base")
        check_cases("functional", self.test_cases)

    def process_cases(self):
        self.preprocess_and_check()
        self.test_cases, self.case_num = normalize_cases(self.test_cases)

        # logger.info(f"case num: {self.case_num}")
