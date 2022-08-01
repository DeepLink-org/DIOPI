import conformance
from conformance.testcase_configs import configs
from conformance.testcase_parse import CaseCollection
from conformance.gen_inputs import GenData


def generate_inputs():
    case_collection = CaseCollection(
        configs=configs
    )
    gen_data = GenData(case_collection)
    gen_data.generate()


if __name__ == "__main__":
    generate_inputs()
