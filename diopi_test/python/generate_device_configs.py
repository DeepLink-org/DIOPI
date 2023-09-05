import sys
import json
from conformance.diopi_configs import diopi_configs

"""

to generate a new device config file for diopi test

usage example:
    python generate_device_configs.py ../../impl/ascend/device_configs.py

"""

if __name__ == "__main__":
    output_file = sys.argv[1]

    tab = " " * 4

    output_str = ""
    output_str += "from .device_config_helper import Skip\n"
    output_str += "from .diopi_runtime import Dtype\n"
    output_str += "\ndevice_configs = {\n"

    for config_name in diopi_configs:
        output_str += tab * 1 + "'{}': dict(\n".format(config_name)
        op_names = diopi_configs[config_name]["name"]
        output_str += tab * 2 + "name={},\n".format(str(op_names))

        if "tensor_para" in diopi_configs[config_name]:
            output_str += tab * 2 + "tensor_para=dict(\n"
            output_str += tab * 3 + "args=[\n" + tab * 4 + "{\n"

            if "ins" in diopi_configs[config_name]["tensor_para"]["args"][0]:
                ins = diopi_configs[config_name]["tensor_para"]["args"][0]["ins"]
                output_str += tab * 5 + "\"ins\": {},\n".format(ins)

                if "dtype" in diopi_configs[config_name]["tensor_para"]["args"][0]:
                    dtype = diopi_configs[config_name]["tensor_para"]["args"][0]["dtype"]
                    output_str += tab * 5 + "\"dtype\": ["
                    for index, value in enumerate(dtype):
                        output_str += "Skip({}),".format(value)
                    output_str += "],\n"

                elif "shape" in diopi_configs[config_name]["tensor_para"]["args"][0]:
                    shape = diopi_configs[config_name]["tensor_para"]["args"][0]["shape"]
                    output_str += tab * 5 + "\"shape\": ["
                    for index, value in enumerate(shape):
                        output_str += "Skip({}),".format(value)
                    output_str += "],\n"

                else:
                    if "dtype" in diopi_configs[config_name]:
                        dtype = diopi_configs[config_name]["dtype"]
                    elif "dtype" in diopi_configs[config_name]["tensor_para"]:
                        dtype = diopi_configs[config_name]["tensor_para"]["dtype"]
                    output_str += tab * 5 + "\"dtype\": ["
                    for index, value in enumerate(dtype):
                        output_str += "Skip({}),".format(value)
                    output_str += "],\n"

            else:
                output_str += tab * 5 + "\"ins\": ['{}'],\n".format("input")
                dtype = diopi_configs[config_name]["tensor_para"]["args"][0]["dtype"]
                output_str += tab * 5 + "\"dtype\": ["
                for index, value in enumerate(dtype):
                    output_str += "Skip({}),".format(value)
                output_str += "],\n"

            output_str += tab * 4 + "},\n" + tab * 3 + "]\n"
            output_str += tab * 2 + "),\n"
        else:
            para = diopi_configs[config_name]["para"]
            output_str += tab * 2 + "para=dict(\n"
            for item_name in para:
                output_str += tab * 3 + "{}=[".format(item_name)
                for value in para[item_name]:
                    output_str += "Skip({}),".format(value)
                output_str += "],\n"

            output_str += tab * 2 + "),\n"
        output_str += tab + "),\n\n"
    output_str += "}\n"

    f = open(output_file, "w")
    f.write(output_str)
    f.close()
