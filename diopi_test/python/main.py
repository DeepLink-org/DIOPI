import subprocess
import argparse
import shlex
import os
import sys

sys.path.append('../python/configs')
from conformance.utils import is_ci, error_counter, write_report
from conformance.utils import logger, nhwc_op, dtype_op, dtype_out_op
from conformance.global_settings import glob_vars
from conformance.model_list import model_list, model_op_list
from conformance.config_parser import ConfigParser
from conformance.gen_input import GenInputData
from conformance.gen_output import GenOutputData
from conformance.collect_case import DeviceConfig, CollectCase


cur_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(cur_dir, 'cache')
data_path = os.path.join(cache_path, 'data')
diopi_case_item_path = os.path.join(cache_path, 'diopi_case_items.cfg')
device_case_item_path = os.path.join(cache_path, 'device_case_items.cfg')

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')
    parser.add_argument('--mode', type=str, default='test',
                        help='running mode, available options: gen_data, run_test and utest')
    parser.add_argument('--fname', type=str, default='all_ops',
                        help='the name of the function for which the test will run (default: all_ops)')
    parser.add_argument('--model_name', type=str, default='',
                        help='Get op list of given model name')
    parser.add_argument('--filter_dtype', type=str, nargs='*',
                        help='The dtype in filter_dtype will not be processed')
    parser.add_argument('--get_model_list', action='store_true',
                        help='Whether return the supported model list')
    parser.add_argument('--nhwc', action='store_true',
                        help='Whether to use nhwc layout for partial tests')
    parser.add_argument('--nhwc_min_dim', type=int, default=3,
                        help='Whether to use nhwc layout for 3-dim Tensor')
    parser.add_argument('--four_bytes', action='store_true',
                        help='Whether to use 4-bytes data type for partial tests')
    parser.add_argument('--impl_folder', type=str, default='',
                        help='folder to find device configs')
    parser.add_argument('--failure_debug_level', type=int, default=0,
                        help='Whether to print debug information when failing the test. 0 for printing nothing, 1 for printing config, 2 for printing config, inputs and outputs')
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
        model_name = args.model_name.lower()
        if args.model_name != '':
            logger.info(f"the op list of {args.model_name}: {model_op_list[model_name]}")
        else:
            from diopi_configs import diopi_configs
            cfg_parse = ConfigParser(diopi_case_item_path)
            cfg_parse.parser(diopi_configs)
            cfg_parse.save()
            GenInputData.run(diopi_case_item_path, os.path.join(data_path, 'inputs'), args.fname)
            GenOutputData.run(diopi_case_item_path, os.path.join(data_path, 'inputs'), os.path.join(data_path, 'outputs'), args.fname)

        if args.impl_folder != '':
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
            coll.save(device_case_item_path)

    elif args.mode == 'run_test':
        import conformance as cf
        cf.ConformanceTest.run(args.fname, args.model_name.lower(), args.filter_dtype, args.failure_debug_level, args.impl_folder)
        write_report()
    elif args.mode == 'utest':
        call = "python3 -m pytest -vx tests"
        subprocess.call(shlex.split(call))  # nosec
    else:
        print("available options for mode: gen_data, run_test and utest")

    # if is_ci != "null" and error_counter[0] != 0:
    #     raise DiopiException(str(error_counter[0]) + " errors during this program")
