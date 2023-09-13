import subprocess
import argparse
import shlex
import os
import sys
import pytest

sys.path.append('../python/configs')
from conformance.op_nhwc import nhwc_op
from conformance.op_four_types import dtype_op, dtype_out_op
from conformance.utils import is_ci, error_counter, write_report
from conformance.utils import logger
from conformance.global_settings import glob_vars
from conformance.model_list import model_list, model_op_list
from configs import model_config
from conformance.config_parser import ConfigParser
from conformance.gen_input import GenInputData
from conformance.gen_output import GenOutputData
from conformance.collect_case import DeviceConfig, CollectCase


cur_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(cur_dir, 'cache')


if not os.path.exists(cache_path):
    os.makedirs(cache_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')

    general_args = parser.add_argument_group('general')
    general_args.add_argument('--mode', type=str, default='test',
                              help='running mode, available options: gen_data, run_test and utest')
    general_args.add_argument('--get_model_list', action='store_true',
                               help='Whether return the supported model list')
    general_args.add_argument('--failure_debug_level', type=int, default=0,
                              help='Whether to print debug information when failing the test. 0 for printing nothing, 1 for printing config, 2 for printing config, inputs and outputs')

    gen_data_args = parser.add_argument_group('gen_data')
    gen_data_args.add_argument('--fname', type=str, default='all_ops',
                               help='the name of the function for which the test will run (default: all_ops)')
    gen_data_args.add_argument('--model_name', type=str, default='',
                               help='Get op list of given model name')
    gen_data_args.add_argument('--impl_folder', type=str, default='',
                               help='impl_folder')

    gen_case_args = parser.add_argument_group('gen_case')
    gen_case_args.add_argument('--nhwc', action='store_true',
                               help='Whether to use nhwc layout for partial tests')
    gen_case_args.add_argument('--nhwc_min_dim', type=int, default=3,
                               help='Whether to use nhwc layout for 3-dim Tensor')
    gen_case_args.add_argument('--four_bytes', action='store_true',
                               help='Whether to use 4-bytes data type for partial tests')
    gen_case_args.add_argument('--cfg_path', type=str, default='./cache/diopi_case_items.cfg',
                               help='case items cfg path')
    gen_case_args.add_argument('--case_output_dir', type=str, default='./gencases/diopi_case',
                               help='pytest case save dir')

    run_test_args = parser.add_argument_group('run_test')
    run_test_args.add_argument('--file_or_dir', type=str,
                               help='pytest case file or dir')
    run_test_args.add_argument('--html_report', action='store_true',
                               help='generate html report')
    run_test_args.add_argument('--filter_dtype', type=str, nargs='*',
                               help='The dtype in filter_dtype will not be processed')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.get_model_list:
        print(f"The supported model_list: {model_list}")

        for model_name in model_op_list.keys():
            op_list = model_op_list[model_name]
            print(f"The model {model_name}'s refrence op_list: {op_list}")
        exit(0)

    if args.filter_dtype:
        dtype_str_list = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
                          'int64', 'uint64', 'float16', 'float32', 'float64', 'bool']
        for dtype in args.filter_dtype:
            if dtype not in dtype_str_list:
                logger.error(f"expect type in {dtype_str_list} but got '{dtype}'")
                exit(0)

    if args.nhwc:
        logger.info(f"The op_list using nhwc layout: {list(nhwc_op.keys())}",)
        glob_vars.set_nhwc()
        glob_vars.set_nhwc_min_dim(args.nhwc_min_dim)

    if args.four_bytes:
        logger.info(f"The op_list using 32-bit type: {list(dtype_op.keys()) + list(dtype_out_op.keys())}")
        glob_vars.set_four_bytes()

    if args.mode == 'gen_data':
        diopi_case_item_file = 'diopi_case_items.cfg'
        device_case_item_file = '%s_case_items.cfg'

        model_name = args.model_name.lower()
        if args.model_name != '':
            logger.info(f"the op list of {args.model_name}: {model_op_list[model_name]}")
            diopi_configs = eval(f"model_config.{model_name}_config")
            diopi_case_item_file = model_name + '_' + diopi_case_item_file
            device_case_item_file = model_name + '_' + device_case_item_file
        else:
            # set a prefix for dat save path like: data/diopi/inputs
            model_name = 'diopi'
            from diopi_configs import diopi_configs
        diopi_case_item_path = os.path.join(cache_path, diopi_case_item_file)
        device_case_item_path = os.path.join(cache_path, device_case_item_file)
        cfg_parse = ConfigParser(diopi_case_item_path)
        cfg_parse.parser(diopi_configs)
        cfg_parse.save()
        inputs_dir = os.path.join(cache_path, 'data/' + model_name + "/inputs")
        outputs_dir = os.path.join(cache_path, 'data/' + model_name + "/outputs")
        GenInputData.run(diopi_case_item_path, inputs_dir, args.fname)
        GenOutputData.run(diopi_case_item_path, inputs_dir, outputs_dir, args.fname)

        if args.impl_folder != '':
            device_name = os.path.basename(args.impl_folder)
            device_config_path = os.path.join(args.impl_folder, "device_configs.py")
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
            coll.save(device_case_item_path % device_name)
    elif args.mode == 'gen_case':
        model_name = args.model_name.lower() if args.model_name != '' else 'diopi'
        from codegen.gen_case import GenConfigTestCase
        if not os.path.exists(args.case_output_dir):
            os.makedirs(args.case_output_dir)
        gctc = GenConfigTestCase(module=model_name, config_path=args.cfg_path, tests_path=args.case_output_dir)
        gctc.gen_test_cases()
    elif args.mode == 'run_test':
        pytest_args = [args.file_or_dir]
        if args.html_report:
            pytest_args.extend(['--report=report.html', '--title=DIOPI Test', '--template=2'])
        pytest.main(pytest_args)
    elif args.mode == 'utest':
        call = "python3 -m pytest -vx tests"
        subprocess.call(shlex.split(call))  # nosec
    else:
        print("available options for mode: gen_data, run_test and utest")

    # if is_ci != "null" and error_counter[0] != 0:
    #     raise DiopiException(str(error_counter[0]) + " errors during this program")
