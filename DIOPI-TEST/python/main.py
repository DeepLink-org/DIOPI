import sys
import conformance as cf
from conformance import functions as F


def cuda_test():
    x = cf.Tensor(size=(2, 3, 5), dtype=cf.int32, device=cf.device("device"))
    x.fill_(10)
    z = F.add(x, x)
    print(x, z)


def generate_inputs():
    case_collection = cf.CaseCollection(
        configs=cf.configs
    )
    gen_data = cf.GenData(case_collection)
    gen_data.generate()


if __name__ == "__main__":
    opt = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if opt == "gen":
        generate_inputs()
    else:
        cuda_test()
