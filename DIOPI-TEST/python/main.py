import subprocess
import argparse
import shlex
import conformance as cf
from conformance.utils import is_ci, error_counter, DiopiException


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')
    parser.add_argument('--mode', type=str, default='test',
                        help='running mode, available options: gen_data, run_test and utest')
    parser.add_argument('--fname', type=str, default='all',
                        help='the name of the function for which the test will run (default: all)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'gen_data':
        cf.GenInputData.run(args.fname)
        cf.GenOutputData.run(args.fname)
    elif args.mode == 'run_test':
        cf.ConformanceTest.run(args.fname)
    elif args.mode == 'utest':
        call = "python3 -m pytest -vx tests"
        subprocess.call(shlex.split(call))  # nosec
    else:
        print("available options for mode: gen_data, run_test and utest")
 
    if is_ci != "null" and error_counter[0] != 0:
        raise DiopiException(str(error_counter[0]) + " errors during this program")
