import argparse
import numpy as np
import conformance as cf
from conformance import functions as F
from conformance import gen_outputs, testcase_run


def cuda_test():
    print("test constructing a cf Tensor and fill it with a value")
    x = cf.Tensor(size=(2, 3, 5), dtype=cf.float32, device=cf.device("device"))
    x.fill_(10)
    z = F.add(x, x)
    print(x, z)

    print("test constrcting a cf Tensor from a numpy ndarray")
    a = np.array([[-1, 2.1], [3, 5.0]], dtype=np.float32)
    b = cf.Tensor.from_numpy(a)
    print(a, b)

    print("test functions relu & softmax on CUDA device")
    c = F.relu(b)
    print(c)
    w = F.softmax(z, 0)
    print(z, w)


def generate_inputs(opname):
    case_collection = cf.CaseCollection(
        configs=cf.configs
    )
    gen_data = cf.GenData(case_collection)
    gen_data.generate(opname)


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')
    parser.add_argument('--mode', type=str, default='test',
        help='running mode, available options: gen_input, gen_output, run & test')
    parser.add_argument('--fn', type=str, default='all',
        help='the name of the function for which the test will run')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'gen_input':
        generate_inputs(args.fn)
    elif args.mode == 'gen_output':
        gen_outputs.generate(args.fn)
    elif args.mode == 'run':
        testcase_run.run(args.fn)
    else:
        cuda_test()
