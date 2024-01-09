import subprocess
import argparse
import shlex
import os
import sys
import pytest

from conformance.global_op_list import nhwc_op, dtype_op, dtype_out_op
from conformance.utils import logger
from conformance.global_settings import glob_vars
from conformance.model_list import model_list, model_op_list
sys.path.append("../python/configs")


cur_dir = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(cur_dir, "cache")


def parse_args():
    parser = argparse.ArgumentParser(description="Conformance Test for DIOPI")

    general_args = parser.add_argument_group("general")
    general_args.add_argument(
        "--mode",
        type=str,
        default="test",
        help="running mode, available options: gen_data, gen_case, run_test and utest",
    )
    general_args.add_argument(
        "--use_db", action="store_true", help="use database to save test data"
    )
    general_args.add_argument(
        "--db_path",
        type=str,
        default="sqlite:///./cache/diopi_testrecord.db",
        help="database path",
    )
    general_args.add_argument(
        "--get_model_list",
        action="store_true",
        help="Whether return the supported model list",
    )
    general_args.add_argument(
        "--failure_debug_level",
        type=int,
        default=0,
        help="Whether to print debug information when failing the test. 0 for printing nothing, 1 for printing config, 2 for printing config, inputs and outputs",
    )

    gen_data_args = parser.add_argument_group("gen_data")
    gen_data_args.add_argument(
        "--fname",
        type=str,
        default="all_ops",
        help="the name of the function for which the test will run (default: all_ops)",
    )
    gen_data_args.add_argument(
        "--model_name", type=str, default="", help="Get op list of given model name"
    )

    gen_case_args = parser.add_argument_group("gen_case")
    gen_case_args.add_argument(
        "--impl_folder", type=str, default="", help="impl_folder"
    )
    gen_case_args.add_argument(
        "--nhwc",
        action="store_true",
        help="Whether to use nhwc layout for partial tests",
    )
    gen_case_args.add_argument(
        "--nhwc_min_dim",
        type=int,
        default=3,
        help="Whether to use nhwc layout for 3-dim Tensor",
    )
    gen_case_args.add_argument(
        "--four_bytes",
        action="store_true",
        help="Whether to use 4-bytes data type for partial tests",
    )
    gen_case_args.add_argument(
        "--case_output_dir",
        type=str,
        default="./gencases",
        help="pytest case save dir",
    )

    run_test_args = parser.add_argument_group("run_test")
    run_test_args.add_argument(
        "--test_cases_path",
        type=str,
        default="",
        help="pytest case file or dir",
    )
    run_test_args.add_argument(
        "--html_report", action="store_true", help="generate html report"
    )
    run_test_args.add_argument("--pytest_args", type=str, help="pytest args", default='')
    run_test_args.add_argument(
        "--filter_dtype",
        type=str,
        nargs="*",
        help="The dtype in filter_dtype will not be processed",
    )
    run_test_args.add_argument(
        "--test_result_path",
        type=str,
        default="report.xlsx",
        help="excel report save path",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    glob_vars.use_db = args.use_db
    from conformance.db_operation import db_conn, BenchMarkCase, DeviceCase

    db_conn.init_db(args.db_path)

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
        logger.info(
            f"The op_list using nhwc layout: {list(nhwc_op.keys())}",
        )
        glob_vars.set_nhwc()
        glob_vars.set_nhwc_min_dim(args.nhwc_min_dim)

    if args.four_bytes:
        logger.info(
            f"The op_list using 32-bit type: {list(dtype_op.keys()) + list(dtype_out_op.keys())}"
        )
        glob_vars.set_four_bytes()

    if args.mode == "gen_data":
        from conformance.gen_data import gen_data
        db_conn.drop_case_table(BenchMarkCase)
        inp_items, outp_items = gen_data(args.model_name.lower(), cache_path,
                                         args.fname)
        db_conn.insert_benchmark_case(inp_items, outp_items)
    elif args.mode == "gen_case":
        from conformance.gen_case import gen_case
        db_conn.drop_case_table(DeviceCase)
        db_case_items = gen_case(cache_path, cur_dir, args.model_name, args.fname, args.impl_folder, args.case_output_dir)
        db_conn.insert_device_case(db_case_items)
    elif args.mode == "run_test":
        if args.test_cases_path == "":
            model_name = args.model_name.lower() if args.model_name else "diopi"
            test_cases_path = os.path.join(args.case_output_dir, model_name + "_case")
        else:
            test_cases_path = args.test_cases_path
        pytest_args = [test_cases_path]
        if args.filter_dtype:
            filter_dtype_str = " and ".join(
                [f"not {dtype}" for dtype in args.filter_dtype]
            )
            pytest_args.append(f"-m {filter_dtype_str}")
        if args.html_report:
            pytest_args.extend(
                ["--report=report.html", "--title=DIOPI Test", "--template=2"]
            )
        if args.test_result_path:
            pytest_args.append(f"--test_result_path={args.test_result_path}")
        if args.pytest_args is not None:
            pytest_args.extend(args.pytest_args.split())
        pytest_args = ['--cache-clear', '--disable-warnings'] + pytest_args
        exit_code = pytest.main(pytest_args)
        if exit_code != 0:
            raise SystemExit(exit_code)
    elif args.mode == "utest":
        call = "python3 -m pytest -vx tests/diopi"
        # FIXME fix ascend utest error
        subprocess.call(shlex.split(call))
        exit_code = subprocess.call(shlex.split(call))  # nosec
        if exit_code != 0:
            raise SystemExit(exit_code)
    elif args.mode == "utest_diopi_test":
        call = "python3 -m pytest -vx tests/diopi_test"
        exit_code = subprocess.call(shlex.split(call))  # nosec
        if exit_code != 0:
            raise SystemExit(exit_code)
    else:
        print("available options for mode: gen_data, run_test and utest")

    # if is_ci != "null" and error_counter[0] != 0:
    #     raise DiopiException(str(error_counter[0]) + " errors during this program")
