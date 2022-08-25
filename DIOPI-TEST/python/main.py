import subprocess
import argparse
import shlex
import conformance as cf
from conformance import testcase_run


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')
    parser.add_argument('--mode', type=str, default='test',
                        help='running mode, available options: gen_input, gen_output, run & utest')
    parser.add_argument('--fn', type=str, default='all',
                        help='the name of the function for which the test will run')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'gen_input':
        case_collection = cf.CaseCollection(configs=cf.configs)
        gen_data = cf.GenInputData(case_collection)
        gen_data.run(args.fn)
    elif args.mode == 'gen_output':
        gen_data = cf.GenOutputData()
        gen_data.run(args.fn)
    elif args.mode == 'run':
        testcase_run.run(args.fn)
    elif args.mode == 'utest':
        call = "python3 -m pytest -vx tests"
        subprocess.call(shlex.split(call))  # nosec
    else:
        print("available options for mode: gen_input, gen_output, run & utest")
