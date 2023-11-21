# Copyright (c) 2023, DeepLink.

import os
import sys
from .utils import logger
from .model_list import model_op_list
from .gen_input import GenInputData
from .gen_output import GenOutputData
from .config_parser import ConfigParser
from .collect_case import DeviceConfig, CollectCase

sys.path.append("../python/configs")


def gen_case(cache_path=".", cur_dir="", model_name="", fname="", impl_folder="", case_output_dir="",  
             diopi_case_item_file="diopi_case_items.cfg", device_case_item_file = "%s_case_items.cfg"):

    if model_name != "":
        logger.info(
            f"the op list of {model_name}: {model_op_list[model_name]}"
        )
        diopi_configs = eval(f"model_config.{model_name}_config")
        diopi_case_item_file = model_name + "_" + diopi_case_item_file
        device_case_item_file = model_name + "_" + device_case_item_file
    else:
        # set a prefix for dat save path like: data/diopi/inputs
        model_name = "diopi"
        from diopi_configs import diopi_configs
    diopi_case_item_path = os.path.join(cache_path, diopi_case_item_file)
    device_case_item_path = os.path.join(cache_path, device_case_item_file)

    cfg_parse = ConfigParser(diopi_case_item_path)
    cfg_parse.parser(diopi_configs, fname)
    cfg_path = diopi_case_item_path

    if impl_folder != "":
        cfg_path = device_case_item_path % os.path.basename(impl_folder)
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

        opt = DeviceConfig(device_configs)
        opt.run()
        coll = CollectCase(cfg_parse.get_config_cases(), opt.rules())
        coll.collect()
        coll.save(cfg_path)

    from codegen.gen_case import GenConfigTestCase

    if not os.path.exists(case_output_dir):
        os.makedirs(case_output_dir)
        
    gctc = GenConfigTestCase(
        module=model_name, config_path=cfg_path, tests_path=case_output_dir
    )
    gctc.gen_test_cases(fname)
    return gctc.db_case_items