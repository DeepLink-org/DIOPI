import sys
import conformance as cf
from conformance import functions as F
from conformance import gen_outputs, testcase_run
import numpy as np


def cuda_test():
    x = cf.Tensor(size=(2, 3, 5), dtype=cf.float32, device=cf.device("device"))
    x.fill_(10)
    z = F.add(x, x)
    print(x, z)
    print('numpy value:\n', z.numpy())

    a = np.array([[-1, 2.1], [3, 5.0]], dtype=np.float32)
    print(a)
    b = cf.Tensor.from_numpy(a)
    print(b)
    c = F.relu(b)
    print(c)
    w = F.softmax(z, 0)
    print(z, w)


def generate_inputs():
    case_collection = cf.CaseCollection(
        configs=cf.configs
    )
    gen_data = cf.GenData(case_collection)
    gen_data.generate()


if __name__ == "__main__":
    opt = sys.argv[1] if len(sys.argv) > 1 else "gen_input"
    if opt == "gen_input":
        generate_inputs()
    elif opt == "gen_output":
        gen_outputs.generate()
    elif opt == "run":
        testcase_run.run()
    else:
        cuda_test()
