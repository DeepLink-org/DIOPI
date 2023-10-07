import pandas as pd
import ast
import re
import argparse
from collections import defaultdict

# param in op_tensor_param must be a tensor, even if it is not defined
op_tensor_param = {
    'diopiConvolution2d': 'bias',
}

# param in op_normal_param must be a normal param, even if it is a tensor
op_normal_param = {
    'diopiMaxPool2dWithIndices': 'return_indices',
}

# param in op_skip_param should be ignored
op_skip_param = {
    'diopiConvolution2d': ['transposed', 'output_padding'],
    'diopiLogSoftmax': ['half_to_float'],
    'diopiSoftmax': ['half_to_float'],
    'diopiNLLLoss': ['total_weight'],
    'diopiNormalInp': ['generator'],
    'diopiUniformInp': ['generator'],
    'diopiIndexPut': ['unsafe'],
    'diopiMax': ['max_indices', 'max'],
    'diopiMin': ['min_indices', 'min'],
    'diopiSort': ['indices', 'values'],
    'diopiTopk': ['indices', 'values'],
    'diopiUnique': ['indices', 'counts'],
    'diopiUpsampleNearest': ['scales_h', 'scales_w'],
    'diopiUpsampleLinear': ['scales_h', 'scales_w'],
}
# param name translation(all of the ops)
param_name_translation = {
    "self": "input",
    "from": "start",
    "to" : "end",
    "output_size": "size",
}
# op param name translation(specific ops)
op_param_name_translation = {
    'diopiIndexPut': {
        "indices": "indices1",
    },
    'diopiMaxPool2dWithIndices': {
        "indices": "return_indices",
    },
}


def extract_sizes(args_str):
    matches = re.findall(r'sizes:\s*\[([\d,\s]*)\]', args_str)

    sizes_list = []
    for match in matches:
        if match.strip():
            sizes_values = match.split(',')
            sizes_list.append(tuple(int(val.strip()) for val in sizes_values))
        else:
            sizes_list.append(tuple())
    return sizes_list if sizes_list else None


def extract_dtype(args_str):
    dtype_matches = re.findall(r'dtype:\s*([\w\s]+)', args_str)
    dtypes = [match.strip() for match in dtype_matches if match.strip()]
    return dtypes if dtypes else None


def extract_requires_grad(args_str):
    requires_grad_matches = re.findall(r'requires_grad:\s*(\w+)', args_str)
    requires_grads = [match == 'true' for match in requires_grad_matches]
    return requires_grads if requires_grad_matches else None


def extract_args(args_str: str, op_name: str) -> dict:
    args_str = re.sub(r"'undefined'|undefined", "None", args_str)
    args_list = ast.literal_eval(args_str)
    # filter out, out1, out2, output... but maintain output_size
    filtered_args = [arg for arg in args_list if not re.search(r'^(out(?=\d|\:)|^output:\[)', arg)]

    result = {}
    for item in filtered_args:
        key, values_str = item.split(':', 1)
        if key in op_skip_param.get(op_name, []):
            continue
        # first scan for normal translation, then fetch the specific one
        key = param_name_translation.get(key, key)
        key = op_param_name_translation.get(op_name, {}).get(key, key)


        sizes = extract_sizes(values_str)
        dtypes = extract_dtype(values_str)
        requires_grads = extract_requires_grad(values_str)

        # Convert tensor structure into dictionary format
        arg_dict = {}
        if sizes:
            arg_dict['shape'] = sizes
        if dtypes:
            arg_dict['dtype'] = dtypes
        if requires_grads is not None:
            arg_dict['requires_grad'] = requires_grads

        if arg_dict:
            if op_normal_param.get(op_name) == key:
                result[key] = True if arg_dict.keys() else False
            else:
                result[key] = arg_dict
        else:
            # Check for known tensor params and handle them separately
            if op_tensor_param.get(op_name) == key:
                result[key] = {'shape': None, 'dtype': None, 'requires_grad': None}
            else:
                values = ast.literal_eval(values_str)
                if len(values) == 1:
                    if re.search(r'\[\d+,\s*\]', values_str):
                        # case when dims: [1, ]
                        result[key] = (values[0],)
                    else:
                        result[key] = values[0]
                elif len(values) > 1:
                    result[key] = tuple(values)
                else:
                    result[key] = None
    return result


def aggregate_rows(group: pd.core.frame.DataFrame) -> str:
    # group = group.iloc[0:10]   # debug usage
    func_name = group['diopi_fun'].iloc[0]
    aggregated_params_dict = defaultdict(list)
        
    for _, row in group.iterrows():
        row_shapes = []  # To collect shapes for this specific row

        # specific operations towards `upsample`
        if 'Upsample' in func_name:
            aggregated_params_dict['mode'].append(
                func_name.split("Upsample", 1)[1].lower())
        for key, value in row['extracted_args'].items():
            if isinstance(value, dict) and 'shape' in value and 'dtype' in value:
                # tensor param
                if key not in aggregated_params_dict:
                    aggregated_params_dict[key] = {'shape': [], 'dtype': [], 'requires_grad': []}
                if key == 'tensors':
                    row_shapes.append(value['shape'])
                else:
                    # TODO: update DIOPI to support random length of indices
                    # Now we just fetch the first one
                    aggregated_params_dict[key]['shape'].extend(
                        value['shape'][:1] if value['shape'] is not None else [None])
                aggregated_params_dict[key]['dtype'].extend(value['dtype'] if value['dtype'] is not None else [None])
                aggregated_params_dict[key]['requires_grad'].extend(
                    value['requires_grad'] if value.get('requires_grad') is not None else [None])
            else:
                # normal param
                aggregated_params_dict[key].append(value)

        # Append the aggregated shapes for 'tensors' key
        if row_shapes and 'tensors' in aggregated_params_dict:
            aggregated_params_dict['tensors']['shape'].extend(row_shapes)

    return f"'{func_name}': {dict(aggregated_params_dict)}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--input', '-i', type=str, default='dipu_ops.csv', help='Input CSV filename')
    parser.add_argument('--output', '-o', type=str, default='cv_configs/resnet50_ops.py', help='Output Python filename')
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=None)

    df = df[~df['diopi_fun'].str.contains('Backward', case=False)]
    df = df[~(df['args'] == '[]')].drop_duplicates()
    df['extracted_args'] = df.apply(lambda row: extract_args(row['args'], row['diopi_fun']), axis=1)

    res = df.groupby('diopi_fun').apply(aggregate_rows).tolist()

    with open(args.output, 'w') as f:
        f.write('{\n')
        for entry in res:
            f.write('    ' + str(entry) + ',\n')
        f.write('}\n')
