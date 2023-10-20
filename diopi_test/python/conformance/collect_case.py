# Copyright (c) 2023, DeepLink.

import os
import copy
import pickle

from skip import Skip


class DeviceConfig(object):
    def __init__(self, cfg: dict = {}) -> None:
        # {
        #   name: dict(
        #       tol: {}
        #       skip: {}    'para', 'tensor_para'
        #   )
        # }
        #
        self._config = cfg
        self._device_config = {}

        self._device_rules = {}

        # self.devcie_tol = {}
        # self.devcie_skip = {}

    def _expand_config_with_name(self):
        for key, value in self._config.items():
            names = value["name"]
            for name in names:
                nkey = "::".join([key, name])
                self._device_config[nkey] = copy.deepcopy(value)
                self._device_config[nkey]["name"] = name

    def _collect_options(self, key, value):
        # avoid reconfig same case
        if key in self._device_rules.keys():
            raise AttributeError(
                f"Device Config Error for {key}: config {key} many times."
            )
        if "name" not in value.keys():
            raise AttributeError(
                f"Device Config Error for {key}: config {key} has not name."
            )
        # cfg_name = '::'.join([key, value['name']])
        cfg_name = key
        self._device_rules[cfg_name] = dict()

        tols = ["atol", "rtol", "atol_half", "rtol_half"]
        for tol in tols:
            if tol in value.keys():
                if "tol" not in self._device_rules[cfg_name].keys():
                    self._device_rules[cfg_name]["tol"] = {}
                self._device_rules[cfg_name]["tol"][tol] = value[tol]

        if "para" in value.keys():
            self._device_rules[cfg_name]["skip"] = {}
            self._device_rules[cfg_name]["skip"]["para"] = {}
            for k, v in value["para"].items():
                self._device_rules[cfg_name]["skip"]["para"][k] = set(
                    [i.value() for i in v]
                )

        if "tensor_para" in value.keys() and "args" in value["tensor_para"].keys():
            if "skip" not in self._device_rules[cfg_name].keys():
                self._device_rules[cfg_name]["skip"] = dict()
            self._device_rules[cfg_name]["skip"][
                "tensor_para"
            ] = {}  # no need args for filter rule
            args_list = value["tensor_para"]["args"]
            expand_arg_list = []
            for arg in args_list:
                ins_name = arg["ins"]
                for name in ins_name:
                    narg = copy.deepcopy(arg)
                    narg["ins"] = name
                    expand_arg_list.append(narg)
            for ins in expand_arg_list:
                ins_key = ins["ins"]
                self._device_rules[cfg_name]["skip"]["tensor_para"][
                    ins_key
                ] = {}  # for each args, i.e. input
                for k, v in ins.items():
                    if k != "ins":
                        self._device_rules[cfg_name]["skip"]["tensor_para"][ins_key][
                            k
                        ] = set()
                        for sk in v:
                            if isinstance(sk, Skip):
                                self._device_rules[cfg_name]["skip"]["tensor_para"][
                                    ins_key
                                ][k].add(sk.value())

    # @staticmethod
    def run(self):
        self._expand_config_with_name()
        for key, val in self._device_config.items():
            self._collect_options(key, val)

    def rules(self):
        return self._device_rules


# filter cases for device: i.e. camb
class CollectCase(object):
    def __init__(self, diopi_cfg: dict, dev_cfg: DeviceConfig) -> None:
        self._diopi_items = diopi_cfg
        self._device_filter = dev_cfg

        self._device_cases = {}

    def _filter_case(self) -> bool:
        # helper function: if True, skip current case
        def _filter(case_key: str, case_cfg: dict, filter_rule: dict):
            case_key = "_".join(case_key.split("_")[:-1])
            if case_key not in filter_rule.keys():
                return False

            rule = filter_rule[case_key]
            # tol
            if "tol" in rule.keys():
                for tol in rule["tol"]:
                    case_cfg[tol] = rule["tol"][tol]

            # skip
            if "skip" not in rule.keys():
                return False

            # 'para'
            if "para" in rule["skip"].keys() and "para" in case_cfg.keys():
                para_case = case_cfg["para"]
                para_rule = rule["skip"]["para"]

                for pk, pv in para_case.items():
                    if pk in para_rule.keys() and pv in para_rule[pk]:
                        return True
            if (
                "tensor_para" in rule["skip"].keys() and "tensor_para" in case_cfg.keys()
            ):
                # 'tensor_para
                tp_case_args = case_cfg["tensor_para"]["args"]  # this is a list
                tp_rule_args = rule["skip"]["tensor_para"]  # this is a dict
                for ins in tp_case_args:
                    if ins["ins"] in tp_rule_args.keys():
                        for fk in tp_rule_args[ins["ins"]].keys():
                            ins_fk_tmp = (
                                ins[fk] if not isinstance(ins[fk], list) else ins[fk][0]
                            )
                            if ins_fk_tmp in tp_rule_args[ins["ins"]][fk]:
                                return True
            return False

        for key, item in self._diopi_items.items():
            if _filter(key, item, self._device_filter):
                # print(f"_filter: {key} : Filtered.")
                continue
            self._device_cases[key] = item

    def collect(self):
        self._filter_case()

    def get_cases(self):
        return self._device_cases

    def save(self, path="../cache/device_case_items.cfg"):
        with open(path, "wb") as fil:
            pickle.dump(self.get_cases(), fil)


if __name__ == "__main__":
    from config_parser import ConfigItem

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    impl_folder = os.path.join(cur_dir, "../../../impl/camb")
    device_config_path = os.path.join(impl_folder, "device_configs.py")

    dst_path = os.path.join(cur_dir, "device_configs.py")

    def unlink_device():
        if os.path.islink(dst_path):
            os.unlink(dst_path)

    unlink_device()
    os.symlink(device_config_path, dst_path)
    import atexit

    atexit.register(unlink_device)

    from device_configs import device_configs

    diopi_cfg_path = "./cache/diopi_case_items.cfg"
    if os.path.isfile(diopi_cfg_path):
        print(True)

    with open(diopi_cfg_path, "rb") as f:
        diopi_configs = pickle.load(f)

    opt = DeviceConfig(device_configs)
    opt.run()

    coll = CollectCase(diopi_configs, opt.rules())
    coll.collect()
    with open("../cache/device_case_items.cfg", "wb") as f:
        pickle.dump(coll.get_cases, f)
