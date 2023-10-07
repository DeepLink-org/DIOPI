# Copyright (c) 2023, DeepLink.
import os
import re
import ast
import argparse
import logging
log_format = '%(asctime)s - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

name_translation = {
    "diopiConvolution2d": "conv2d",
    "diopiCrossEntropyLoss": "cross_entropy",
    "diopiMaxPool2dWithIndices": "max_pool2d",
    "diopiFill": "fill_",
    "diopiMaxAll": "max",
    "diopiMinAll": "min",
    "diopiUpsampleNearest": "interpolate",
    "diopiUpsampleLinear": "interpolate",
}
param_type_translation = {
    "training": bool,
    "ceil_mode": bool,
    "keepdim": bool,
    "reduction": lambda x: "none" if x == 0 else ("mean" if x == 1 else "sum"),
    "dim": lambda x: None if isinstance(x, list) and len(x) == 0 else x,
    "accumulate": bool,
    "descending": bool,
    "largest": bool,
    "sorted": bool,
    "return_inverse": bool,
    "return_counts": bool,
}
func_interface = {
    'torch': ['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or', 'cat',
              'stack', 'flip', 'mean', 'fill', 'sum', 'abs', 'all',
              'any', 'arange', 'bitwise_and', 'bitwise_or', 'clamp',
              'clamp_min', 'clamp_max', 'exp', 'floor', 'neg',
              'log', 'log2', 'log10', 'minimum', 'maximum', 'unique',
              'max', 'min', 'nonzero', 'sgn', 'sort', 'topk',
              'cos', 'erf', 'erfinv', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
    'torch.nn.functional': ['conv2d', 'batch_norm'],
    'torch.Tensor': ['normal_', 'fill_', 'repeat'],
    "CustomizedTest": ['index', 'index_put']
}
no_output_ref = ['randperm', 'uniform', 'dropout', 'dropout2d', 'normal', 'multinomial', 'normal_']
saved_args = {"sigmoid": "0", 'softmax': '0', 'log_softmax': '0', 'tanh': '0', 'cholesky_ex': '0', 'cdist': '0',
              'triangular_solve': '0'}
# For some ops, only some of the params that need to requires_grad
requires_backward = {'cholesky_ex': '0', 'max_pool2d': '0'}
# For some ops that doesn't have diopiBackward function, skip requires_grad
skip_backward_ops = ['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
                     'lt', 'gt', 'ge', 'logical_and', 'logical_or', 'cat', 'stack', 'flip', 'mean', 'fill_', 'sum', 'relu', 'normal_', 'topk', 'abs', 'neg', 'sgn',
                     'uniform']
# For some ops that doesn't need to use is_inplace
skip_inplace_ops = ['normal_', 'fill_']

atol_rtol = {
    'batch_norm': dict(atol=1e-1, rtol=1e-2, atol_half=1e-1, rtol_half=1e-2),
    'conv2d': dict(atol=1e-3, rtol=1e-3),
    'linear': dict(atol=1e-3, rtol=1e-4, atol_half=1e-1, rtol_half=1e-2,),
    'clamp': dict(atol=1e-4, rtol=1e-5,)
}

tensor_indent = "                    "
para_indent = "            "
key_indent = "        "
seq_name = ['cat', 'stack']

dtype_mappings = {
    'float': ('[Dtype.float32]', 'Genfunc.randn'),
    'double': ('[Dtype.float32]', 'Genfunc.randn'),
    'long int': ('[Dtype.int64]', 'Genfunc.randint'),
    'bool': ('[Dtype.bool]', 'Genfunc.mask'),
    'int': ('[Dtype.int32]', 'Genfunc.randint'),
    'unsigned char': ('[Dtype.uint8]', 'Genfunc.randint'),
    'char': ('[Dtype.int8]', 'Genfunc.randn'),
    'half': ('[Dtype.float16]', 'Genfunc.randn'),
    'short': ('[Dtype.int16]', 'Genfunc.randn'),
    'complex<float>': ('[Dtype.complex64]', 'Genfunc.randn_cmplx'),
    'complex<double>': ('[Dtype.complex128]', 'Genfunc.randn_cmplx')
}

gen_func = {
    'cholesky_ex:input': 'Genfunc.sym_mat',
    'normal:std': 'Genfunc.positive',
    'adadelta:square_avg': 'Genfunc.positive',
    'adadelta:acc_delta': 'Genfunc.positive',
    'rsqrt:input': 'Genfunc.positive',
    'multinomial:input': 'Genfunc.positive',
    'batch_norm:running_var': 'Genfunc.positive',
    'adamw:exp_avg_sq': 'Genfunc.positive',
    'adam:exp_avg_sq': 'Genfunc.positive',
    'erfinv:input': 'dict(fn=Genfunc.uniform, low=-1, high=1)',
    'sqrt:input': 'Genfunc.positive',
    'log:input': 'Genfunc.positive',
    'log2:input': 'Genfunc.positive',
    'log10:input': 'Genfunc.positive'
}

warning_list = ['index_put', 'index']

def convert_op_name(op):
    def camel_to_snake(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    if op in name_translation:
        return name_translation[op]

    op = op.replace("diopi", "")
    op = op.replace("Scalar", "")
    if op.endswith("Inp"):
        op = op[:-3] + "_"
    return camel_to_snake(op)


def get_interface(op_name: str) -> str:
    for interface, ops in func_interface.items():
        if op_name in ops:
            return interface
    return 'torch.nn.functional'  # default


def gen_config_code(contents: dict, file_name: str) -> None:
    names = {}
    assert isinstance(contents, dict), "file content is not a dict!"

    os.system(f"rm -f {file_name}")
    with open(f'{file_name}', 'a') as f:
        f.write("from ...config import Genfunc\n")
        f.write("from ...diopi_runtime import Dtype\n\n")
        f.write(file_name[file_name.find("/") + 1:].rstrip('.py') + " = {\n")

        for name, params in contents.items():
            name = convert_op_name(name)
            para = []
            tensor_para = []

            inplace_flag = False
            if name.endswith('_') and name not in skip_inplace_ops:
                # handle inplace op
                inplace_flag = True
                name = name.rstrip('_')

            if names.get(name):
                names[name] += 1
                config = ["    '" + name + "_case_" + str(names[name]) + "': dict(\n"]
            else:
                config = ["    '" + name + "': dict(\n"]
                names.update({name: 1})
            config.append(key_indent + 'name=["' + name + '"],\n')

            if name in warning_list:
                logging.warning(f"Need to check manually for `{name}` due to randomness or current limitation.")
                # continue    # debug usage
            if name in atol_rtol:
                for k, v in atol_rtol[name].items():
                    config.append(key_indent + f'{k}={v:.0e},\n')
            if name in no_output_ref:
                config.append(key_indent + 'no_output_ref=True,\n')
            config.append(key_indent + 'interface=["' + get_interface(name) + '"],\n')
            if inplace_flag:
                config.append(key_indent + 'is_inplace=[True],\n')
            if name in saved_args.keys():
                config.append(key_indent + 'saved_args=dict(output=' + saved_args[name] + '),\n')
            if name in requires_backward.keys():
                config.append(key_indent + 'requires_backward=[' + requires_backward[name] + '],\n')

            for k, v in params.items():
                if k in param_type_translation:
                    v = list(map(param_type_translation[k], v))
                is_tensor = True if isinstance(v, dict) else False
                if is_tensor:
                    tensor_para.append(para_indent + "    {\n" + tensor_indent + '"ins": ["' + str(k) + '"],\n')
                    # Currently, if there is a True in list, we use True
                    if name not in skip_backward_ops:
                        requires_grad = True in set(v["requires_grad"])
                        tensor_para.append(tensor_indent + '"requires_grad":[' + str(requires_grad) + '],\n')
                    tensor_para.append(tensor_indent + '"shape": ' + str(v["shape"]) + ",\n")
                    unique_dtypes = set(v['dtype']) - {None}
                    dtype = next(iter(unique_dtypes)) if unique_dtypes else None
                    if dtype is not None:
                        # TODO: currently we only support generate one dtype (update diopi)
                        assert dtype in dtype_mappings, "unexpected input!"
                        dtype_str, gen_fn_default = dtype_mappings[dtype]
                        gen_fn = gen_func.get(f"{name}:{k}", gen_fn_default)
                        tensor_para.append(tensor_indent + '"dtype": ' + dtype_str + ',\n')
                        tensor_para.append(tensor_indent + '"gen_fn": ' + gen_fn + ',\n')
                    tensor_para.append(para_indent + "    },\n")

                else:
                    para.append(para_indent + str(k) + "=" + str(v) + ",\n")

            if para:
                config.append(key_indent + "para=dict(\n")
                for e in para:
                    config.append(e)
                config.append(key_indent + "),\n")
            if tensor_para:
                config.append(key_indent + "tensor_para=dict(\n")
                config.append(para_indent + "args=[\n")
                for e in tensor_para:
                    config.append(e)
                config.append(para_indent + "],\n")
                if name in seq_name:
                    config.append(para_indent + "seq_name='tensors',\n")
                config.append(key_indent + "),\n")
            config.append("    ),\n")
            config.append("\n")
            for row in config:
                f.write(row)
        f.write("}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate configuration code.")

    parser.add_argument('--input_file', '-i', type=str, default='cv_configs/resnet50_ops.py',
                        help='Input filename containing the operations')

    parser.add_argument('--output_file', '-o', type=str, default='cv_configs/resnet50_config.py',
                        help='Output filename for the generated configuration')

    args = parser.parse_args()

    with open(args.input_file) as f:
        gen_config_code(ast.literal_eval(f.read()), args.output_file)
