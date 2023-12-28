import os
import sys
import pytest


def run_test(test_cases_path="", case_output_dir="", model_name="", fname="all_ops",
             test_result_path='report.xlsx', filter_dtype=None, html_report=False, pytest_args=""):
    if test_cases_path == "":
        model_name = model_name.lower() if model_name else "diopi"
        test_cases_path = os.path.join(case_output_dir, model_name + "_case")
    else:
        test_cases_path = test_cases_path
    args = [test_cases_path]
    if fname and fname != 'all_ops':
        args.append(f'-k {fname}')
    if filter_dtype:
        filter_dtype_str = " and ".join(
            [f"not {dtype}" for dtype in filter_dtype]
        )
        args.append(f"-m {filter_dtype_str}")
    # need pytest-testreport plugin to generate HTML report
    if html_report:
        args.extend(
            ["--report=report.html", "--title=DIOPI Test", "--template=2"]
        )
    args.append(f"--test_result_path={test_result_path}")
    args.extend(pytest_args.split())
    args = ['--cache-clear', '--disable-warnings'] + args
    return pytest.main(args)
