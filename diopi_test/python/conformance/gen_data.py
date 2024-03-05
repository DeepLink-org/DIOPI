# Copyright (c) 2023, DeepLink.

import os
import sys
import shutil
from .utils import logger
from configs import model_config
from .model_list import model_op_list
from .gen_input import GenInputData
from .gen_output import GenOutputData
from .config_parser import ConfigParser
from .exception import GenDataFailedException

sys.path.append("../python/configs")


def gen_data(model_name: str = "", cache_path=".", fname=[""], diopi_case_item_file="diopi_case_items.cfg"):
    model_name = model_name.lower()
    if model_name != "":
        logger.info(
            f"the op list of {model_name}: {model_op_list[model_name]}")
        diopi_configs = eval(f"model_config.{model_name}_config")
        diopi_case_item_file = model_name + "_" + diopi_case_item_file
    else:
        # set a prefix for dat save path like: data/diopi/inputs
        model_name = "diopi"
        from diopi_configs import diopi_configs
    data_path = os.path.join(cache_path, model_name)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    diopi_case_item_path = os.path.join(data_path, diopi_case_item_file)
    cfg_parse = ConfigParser(diopi_case_item_path)
    cfg_parse.parser(diopi_configs, fname)
    cfg_parse.save()
    inputs_dir = os.path.join(data_path + "/inputs")
    outputs_dir = os.path.join(data_path + "/outputs")

    GenInputData.run(diopi_case_item_path, inputs_dir, fname, model_name)
    GenOutputData.run(diopi_case_item_path, inputs_dir, outputs_dir, fname,
                      model_name)
    if GenInputData.err_case_counter > 0 or GenOutputData.err_case_counter > 0:
        raise GenDataFailedException(f"Data generation failed, with {GenInputData.err_case_counter} input case failures and {GenOutputData.err_case_counter} output case failures.")
    return GenInputData.db_case_items, GenOutputData.db_case_items
