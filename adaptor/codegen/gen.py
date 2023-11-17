# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-
import argparse
import os
import re
import yaml
from typing import Tuple, Iterator, List

from filemanager import FileManager
from op_template import OpTemplate as OT
from code_template import CodeTemplate


str_to_diopi_dtype = {
    'uint64': 'diopiDtype_t::diopi_dtype_uint64',
    'int64': 'diopiDtype_t::diopi_dtype_int64',
    'uint32': 'diopiDtype_t::diopi_dtype_uint32',
    'int32': 'diopiDtype_t::diopi_dtype_int32',
    'uint16': 'diopiDtype_t::diopi_dtype_uint16',
    'int16': 'diopiDtype_t::diopi_dtype_int16',
    'uint16': 'diopiDtype_t::diopi_dtype_uint16',
    'int16': 'diopiDtype_t::diopi_dtype_int16',
    'uint8': 'diopiDtype_t::diopi_dtype_uint8',
    'int8': 'diopiDtype_t::diopi_dtype_int8',
    'float64': 'diopiDtype_t::diopi_dtype_float64',
    'float32': 'diopiDtype_t::diopi_dtype_float32',
    'float16': 'diopiDtype_t::diopi_dtype_float16',
    'bool': 'diopiDtype_t::diopi_dtype_bool',
    'complex32': 'diopi_dtype_complex32',
    'complex64': 'diopi_dtype_complex64',
    'complex128': 'diopi_dtype_complex128',
}

str_to_diopi_format = {
    'NCHW': 'diopiMemoryFormat_t::Contiguous',
    'NCL': 'diopiMemoryFormat_t::Contiguous',
    'NLC': 'diopiMemoryFormat_t::ChannelsLast1d',
    'NHWC': 'diopiMemoryFormat_t::ChannelsLast',
    'NDHWC': 'diopiMemoryFormat_t::ChannelsLast3d'
}


default_cast_dtype = {
    'int64': 'int32',
    'float64': 'float32',
    'complex128': 'complex64'
}

cast_strategy = {
    'Default': {
        'int64': 'int32',
        'float64': 'float32',
        'complex128': 'complex64'
    },

    'CastFloatOnly': {
        'float64': 'float32'
    },

    'LogicOp': {
        'int64': 'int32',
        'float64': 'int32'
    }
}


exclude_ops = ['CopyInp', 'CastDtype']
inp_config = {
    'BatchNorm': ['running_mean', 'running_var'],
    'IndexPut': ['out'],
    'Adadelta': ['input', 'grad', 'square_avg', 'acc_delta'],
    'IndexBackward': ['zeros_like_input']
}


def findAllFile(base: str) -> Iterator[str]:
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


# search all the diopi func definitions under the directory dir
def obtain_impl_func(dir: str) -> dict:
    impl_functions = {}
    pattern = r'(?:^|\n)\s*(?:DIOPI_API)?\s*diopiError_t\s+(diopi\w+)\(([^)]*)\)\s*{'
    pattern = re.compile(pattern)
    files = findAllFile(dir)
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            c_code = file.read()

        matches = re.findall(pattern, c_code)

        for match in matches:
            func_name, arguments = match
            arguments = arguments.strip().split(',')
            args_after_fmt = []
            for arg in arguments:
                arg = arg.strip()  # .replace("\r", "").replace("\n", "").replace("\t", "")
                if (arg.strip().startswith('//')):
                    continue
                args_after_fmt.append(arg)
            if func_name not in impl_functions.keys():
                impl_functions[func_name] = {
                    "func_name": func_name,
                    "args": args_after_fmt,
                    "return_type": "diopiError_t"
                }

    return impl_functions


# get the func declararion in the file_path
def obtain_func_declaration(file_path):

    decl_functions = {}
    with open(file_path, 'r') as file:
        c_code = file.read()

    pattern = r'(?:^|\n)\s*DIOPI_API\s+(\w+)\s+(\w+)\(([^)]*)\);'
    pattern = re.compile(pattern)
    matches = re.findall(pattern, c_code)

    for match in matches:
        comment, return_type, func_name, arguments = match
        # skip the comment
        if comment and (comment.strip().startswith('/*') or comment.strip().startswith('//')):
            continue

        arguments = arguments.strip().split(',')

        print("comment:", comment)
        print("Function Name:", func_name)
        print("Return Type:", return_type)
        print("Arguments:")
        args_after_fmt = []
        for arg in arguments:
            arg = arg.strip()
            if (arg.startswith('//')):
                continue
            args_after_fmt.append(arg.strip())
        if func_name not in decl_functions.keys():
            decl_functions[func_name] = {
                "func_name": func_name, "args": args_after_fmt, "return_type": return_type}
    return decl_functions


def prepare() -> Tuple[dict, str]:
    parser = argparse.ArgumentParser(
        description='Generate DIOPI adaptor source files')
    parser.add_argument(
        '-d',
        '--diopi_dir',
        help='path of dependence used to generate code',
        default='../')
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output a list of source files into the given directory',
        default='./')
    parser.add_argument(
        '-c',
        '--config_device',
        help='name of file which contains configs of device',
        default='torch')

    options = parser.parse_args()
    source = os.path.join(options.diopi_dir, 'proto/include/diopi')
    config_path = os.path.join(
        options.diopi_dir, 'impl/', options.config_device)
    device = 'cuda' if options.config_device == 'torch' else options.config_device

    def create_if_not_exist(name):
        if not os.path.exists(name):
            os.makedirs(name)

    create_if_not_exist(options.output_dir)
    dirs = dict(source=source,
                output_dir=options.output_dir,
                config_path=config_path)
    return dirs, device


def get_func_info(content: list) -> Tuple[list, list, list, dict]:
    args = []
    ins = []
    outs = []
    ins_v = {}
    for row in content:
        row = row.replace('\n', '').replace('(', '').replace(');', '')
        args.extend(row.split(','))
    args = [arg.rstrip(' ').lstrip(' ') for arg in args if arg != '']
    for i, arg in enumerate(args):
        tensor_name = arg.split(' ')[1]
        if 'diopiTensorHandle_t*' in arg:
            return None, None, args, None
        if arg.startswith('diopiTensorHandle_t'):
            outs.append(tensor_name)
        elif arg.startswith('diopiConstTensorHandle_t'):
            ins.append(tensor_name)
            if '*' in arg:
                ins_v[tensor_name] = args[i + 1].split(' ')[1]
    return ins, outs, args, ins_v


def get_functions_support(source_dir: str) -> Tuple[dict, dict]:
    with open(os.path.join(source_dir, 'functions.h'), 'r', encoding='utf8')as f:
        content = f.readlines()
    funcs_info = {}
    func_dtypes = []
    param_dtypes = {}
    funcs_decl = {}
    func_name = ''
    sa_func = None
    for idx, row in enumerate(content):
        if row.startswith(' * type'):
            assert len(func_dtypes) == 0
            r = re.match(r' \* type *= *\[(.*)\].*', row)
            func_dtypes = r.group(1).replace(' ', '').split(',')
        elif row.startswith(' * @param'):
            if 'type' in row:
                row_new = row
            elif not content[idx + 1].startswith(' */') and not content[idx + 1].startswith(' * @'):
                row_new = content[idx + 1]
            else:
                continue
            row = row[9:]
            row = row[row.find(' '):].lstrip(' ')
            param_name = row.split(' ')[0]
            r = re.match(r'.*type *= *\[(.*)\].*', row_new)
            if not r:
                r = re.match(r'.*type *= *\[(.*)', row_new)
                if not r:
                    continue
                row1 = content[idx + 1]
                idx2 = row1.find("]")
                dtypes_str = r.group(1) + row1[:idx2].lstrip()
            else:
                dtypes_str = r.group(1)
            dtypes = dtypes_str.replace(' ', '').split(',')
            param_dtypes[param_name] = dtypes
        elif row.startswith(' * @sa'):
            sa_func = row[row.find('diopi'):row.find('(')]
        elif row.startswith("DIOPI"):
            temp_content = []
            idx1 = row.find("(")
            idx0 = row.rfind(" ", 0, idx1)
            func_name = row[idx0 + 1: idx1]
            temp_content.append(row[idx1:-1])
            idx2 = row.find(")")
            while idx2 == -1:
                row1 = content[idx + 1]
                idx2 = row1.find(")")
                temp_content.append(row1.lstrip())
                idx += 1
            if row.startswith("DIOPI_RT_API"):
                continue
            else:
                ins, outs, args, ins_v = get_func_info(temp_content)
                func_decl = 'diopiError_t ' + func_name + \
                    ' '.join(temp_content).replace('\n', '') + '\n\n'
                funcs_decl[func_name] = func_decl
            if func_name not in funcs_info.keys():
                funcs_info[func_name] = {}
            funcs_info[func_name]['call_args'] = args
            if ins is None:
                continue
            if ins_v:
                funcs_info[func_name]['ins_vector'] = ins_v
            funcs_info[func_name]['ins'] = {}
            funcs_info[func_name]['outs'] = {}

            def insert(type, func_dtypes, tensor, param_dtypes):
                if tensor not in param_dtypes.keys():
                    funcs_info[func_name][type][tensor] = func_dtypes
                else:
                    funcs_info[func_name][type][tensor] = param_dtypes[tensor]
            for tensor in ins:
                insert('ins', func_dtypes, tensor, param_dtypes)
            for tensor in outs:
                insert('outs', func_dtypes, tensor, param_dtypes)
            if sa_func:
                funcs_info[func_name]['sa'] = sa_func
            func_dtypes = []
            param_dtypes = {}
            func_name = ''
            sa_func = None
    for func in funcs_info:
        if 'sa' in funcs_info[func].keys():
            assert funcs_info[func]['sa'] in funcs_info.keys()
            from_func_info = funcs_info[funcs_info[func]['sa']]
            for i in from_func_info['ins']:
                if i in funcs_info[func]['ins']:
                    funcs_info[func]['ins'][i] = from_func_info['ins'][i]
                elif i in funcs_info[func]['outs']:
                    funcs_info[func]['outs'][i] = from_func_info['ins'][i]
                if 'grad_' + i in funcs_info[func]['outs']:
                    funcs_info[func]['outs']['grad_' +
                                             i] = from_func_info['ins'][i]
            for out in from_func_info['outs']:
                if out in funcs_info[func]['ins']:
                    funcs_info[func]['ins'][out] = from_func_info['outs'][out]
                elif out in funcs_info[func]['outs']:
                    funcs_info[func]['outs'][out] = from_func_info['outs'][out]
                if 'grad_' + out in funcs_info[func]['ins']:
                    funcs_info[func]['ins']['grad_' +
                                            out] = from_func_info['outs'][out]
    return funcs_info, funcs_decl


def deal_dtype(op_name: str, dtype_config: str, func_infos: dict, tensor_name: str = None):
    strategy = {}
    dtype_configs = dtype_config.replace(' ', '').split(',')
    for idx in range(len(dtype_configs)):
        if not dtype_configs[idx].startswith('('):
            continue
        config = dtype_configs[idx]
        while '->' not in config:
            idx += 1
            config += ',' + dtype_configs[idx]
        r = re.match(r'\((.*)\)->(.*)', config)
        if len(r.groups()) != 2:
            raise ValueError(f"Invalid dtype configuration: {config}")
        else:
            from_dtypes = r.group(1).replace(' ', '').split(',')
            to_dtype = r.group(2)
        for f in from_dtypes:
            # TODO(xintian) : delete '#' when all functions has note
            # assert (op_name == 'Common' or not tensor_name) or \
            #         (tensor_name in func_infos[op_name]['ins'].keys() and f in func_infos[op_name]['ins'][tensor_name]) or \
            #         (tensor_name in func_infos[op_name]['outs'].keys())
            strategy[f.lower()] = to_dtype.lower()
    for s in cast_strategy:
        if len(cast_strategy[s].keys()) != len(strategy.keys()):
            continue
        in_strategy = True
        for d in strategy:
            if d not in cast_strategy[s].keys() or cast_strategy[s][d] != strategy[d]:
                in_strategy = False
                break
        if in_strategy:
            return s
    if tensor_name:
        strategy_name = op_name + tensor_name.capitalize() + 'Cast'
    else:
        strategy_name = op_name + 'Cast'
    cast_strategy[strategy_name] = strategy
    return strategy_name


def analysis_configs(config: List[dict], funcs_info: dict) -> dict:
    if not config:
        return {}
    common_cast = ''
    common_layout = []
    common_contiguous = False
    op_dict = {}
    for info in config:
        [(op_name, op_cfg)] = info.items()
        if op_name == 'common_config':
            if 'dtype' in op_cfg.keys():
                common_cast = deal_dtype(
                    'Common', op_cfg['dtype'], funcs_info)
            if 'layout' in op_cfg.keys():
                common_layout = [op_cfg['layout']] if isinstance(
                    op_cfg['layout'], str) else op_cfg['layout']
            if 'contiguous' in op_cfg.keys():
                common_contiguous = op_cfg['contiguous']
            op_dict['Common'] = {
                'cast': common_cast,
                'layout': common_layout,
                'contiguous': common_contiguous,
            }
        else:
            op_cast = None
            op_tensor = {}
            op_dict[op_name] = {}
            op_layouts = []
            assert op_name in funcs_info
            if 'dtype' in op_cfg.keys():
                op_cast = deal_dtype(op_name, op_cfg['dtype'], funcs_info)
                op_dict[op_name]['cast'] = op_cast
            if 'tensor_dtype' in op_cfg.keys():
                for tensor in op_cfg['tensor_dtype']:
                    assert tensor in funcs_info[op_name]['ins'].keys(
                    ) or tensor in funcs_info[op_name]['outs'].keys()
                    tensor_cast = deal_dtype(
                        op_name, op_cfg['tensor_dtype'][tensor], funcs_info, tensor)
                    op_tensor[tensor] = {}
                    op_tensor[tensor]['cast'] = tensor_cast
            if 'layout' in op_cfg.keys():
                layouts = op_cfg['layout'].replace(' ', '').split(',')
                for layout in layouts:
                    if layout == '':
                        continue
                    if layout == 'NHWC' or layout == 'NCHW' or layout == 'NLC' or layout == 'NCL' or layout == 'NDHWC' or layout == 'NCDHW':
                        op_layouts.append(layout)
                    else:
                        r = re.match(r'(.*)\((.*)\)', layout)
                        tensor_name = r.group(1)
                        tensor_layout = r.group(2)
                        if tensor_name not in op_tensor.keys():
                            op_tensor[tensor_name] = {}
                            op_tensor[tensor_name]['layout'] = tensor_layout
                op_dict[op_name]['layout'] = op_layouts
            if 'contiguous' in op_cfg.keys():
                contiguous_tensor = op_cfg['contiguous'].replace(
                    ' ', '').split(',')
                for tensor_name in contiguous_tensor:
                    if tensor_name not in op_tensor.keys():
                        op_tensor[tensor_name] = {}
                    op_tensor[tensor_name]['contiguous'] = True
            if 'supportComposite' in op_cfg.keys():
                op_dict[op_name]['supportComposite'] = True
            for tensor in list(funcs_info[op_name]['ins'].keys()) + list(funcs_info[op_name]['outs'].keys()):
                if tensor not in op_tensor.keys():
                    op_tensor[tensor] = {}
                if 'cast' not in op_tensor[tensor]:
                    op_tensor[tensor]['cast'] = op_cast if op_cast else common_cast
                if 'contiguous' not in op_tensor[tensor]:
                    op_tensor[tensor]['contiguous'] = True if common_contiguous else False
                if 'layout' not in op_tensor[tensor]:
                    op_tensor[tensor]['layout'] = op_layouts if len(
                        op_layouts) else common_layout
            op_dict[op_name]['tensor'] = op_tensor
    return op_dict


def autogen_cast_strategy():
    cast_code = []

    for strategy in cast_strategy:
        cases = []
        for dtype in cast_strategy[strategy]:
            cases.append('case {from_dtype}:\n \
    convert = true;\n \
    targetDtype = {to_dtype};\n \
    break;'.format(from_dtype=str_to_diopi_dtype[dtype], to_dtype=str_to_diopi_dtype[cast_strategy[strategy][dtype]]))
        cast_code.append(OT.cast_strategy_template.substitute(
            env=dict(cast_name=strategy, cases=cases)))
    return cast_code


def memory_format_to_str(memory_format):
    if len(memory_format) == 0:
        return ', {}'
    memory_format = [format.strip(' ') for format in memory_format]

    formats = []
    for format in memory_format:
        formats.append(str_to_diopi_format[format])
    return ', std::vector<diopiMemoryFormat_t>{' + ','.join(formats) + '}'


def autogen_op_adaptor(op_configs: dict, device: str, func_infos: dict,
                       impl_funcs: dict) -> list:
    adaptors_code = []
    cast = op_configs['Common']['cast'] if 'Common' in op_configs.keys() else ''
    layout = op_configs['Common']['layout'] if 'Common' in op_configs.keys() else [
    ]
    for func in func_infos:
        device_mapping = ''
        op_name = func.lstrip('diopi')
        if func not in impl_funcs:
            if op_configs.get(func, {}).get('supportComposite'):
                device_mapping = 'composite'
            else:
                continue
        if (func not in op_configs.keys() and 'Common' not in op_configs.keys()) or len(list(func_infos[func].keys())) == 1 or op_name in exclude_ops:
            call_args = [arg.split(' ')[-1]
                         for arg in func_infos[func]['call_args']]
            adaptors_code.append(
                OT.adaptor_template.substitute(env=dict(op_name=op_name, attrs=func_infos[func]['call_args'], device=device if not device_mapping else device_mapping,
                                                        new_input='', cast_input='', cast_output='', func_name=func, call_func=func + '(' + ', '.join(call_args) + ')')))
        else:
            op_config = op_configs[func] if func in op_configs.keys() else None
            new_ins = []
            cast_ins = []
            cast_outs = []
            new_input = []
            for tensor in list(func_infos[func]['ins'].keys()) + list(func_infos[func]['outs'].keys()):
                tensor_info = op_config['tensor'][tensor] if op_config else None
                cast_method = tensor_info['cast'] if tensor_info else cast
                memory_format = tensor_info['layout'] if tensor_info else layout
                format_str = memory_format_to_str(memory_format)
                ins = func_infos[func]['ins']
                if tensor in ins:
                    if 'ins_vector' in func_infos[func].keys() and tensor in func_infos[func]['ins_vector'].keys():
                        new_ins_vector_template = CodeTemplate("""\
std::vector<diopiConstTensorHandle_t> ${newinput}(${num}, diopiConstTensorHandle_t());
for (int i = 0; i < ${num}; ++i) {
    castImpl<diopiConstTensorHandle_t${cast}>(ctx, ${input}[i], &${newinput}[i]${memory_format});
}
""")
                        new_input.append(new_ins_vector_template.substitute(env=dict(input=tensor, newinput='new' + tensor.capitalize(), num=func_infos[func]['ins_vector'][tensor],
                                                                                     cast=', ' + cast_method if cast_method else '', memory_format=format_str)))
                    else:
                        new_in = 'new' + tensor.capitalize()
                        new_ins.append(new_in)
                        cast_impl = 'castImpl<diopiConstTensorHandle_t{cast}>(ctx, {tensor}, &{new_tensor}{memory_format});'.format(
                                    cast=', ' + cast_method if cast_method else '', memory_format=format_str, tensor=tensor, new_tensor=new_in)
                        cast_ins.append(cast_impl)
                outs = func_infos[func]['outs']
                if tensor in outs:
                    cast_impl = 'DiopiTensorWrapper<{cast}> {tensor}Wrapper(ctx, {tensor}{memory_format}, {inp});'.format(
                                cast=cast_method, memory_format=format_str, tensor=tensor, inp='true' if ('Inp' in op_name or tensor in inp_config.get(op_name, [])) else 'false')
                    cast_outs.append(cast_impl)
            new_input.append('diopiConstTensorHandle_t ' +
                             ', '.join(new_ins) + ';') if len(new_ins) else ''
            call_args = []
            for arg in func_infos[func]['call_args']:
                name = arg.split(' ')[-1]
                if 'ins_vector' in func_infos[func].keys() and name in func_infos[func]['ins_vector'].keys():
                    new_name = 'new' + name.capitalize() + '.data()'
                elif name in ins.keys():
                    new_name = 'new' + name.capitalize()
                elif name in outs.keys():
                    new_name = name + 'Wrapper'
                else:
                    new_name = name
                call_args.append(new_name)
            adaptors_code.append(
                OT.adaptor_template.substitute(
                    env=dict(
                        op_name=op_name,
                        attrs=', '.join(func_infos[func]['call_args']),
                        device=device_mapping if device_mapping else device,
                        new_input=new_input,
                        cast_input=cast_ins,
                        cast_output=cast_outs,
                        func_name=func,
                        call_func=func + '(' + ', '.join(call_args) + ')'
                    )
                )
            )
    return adaptors_code


def get_impl_funcs_declaration(funcs_decl_raw: dict, funcs_info: dict,
                               impl_funcs: dict) -> dict:
    funcs_decl: dict = {}
    for func in funcs_info.keys():
        if func in impl_funcs:
            funcs_decl[func] = funcs_decl_raw[func]
    return funcs_decl


def get_composite_funcs_declaration(funcs_decl_raw: dict,
                                    funcs_info: dict, impl_funcs: dict,
                                    op_configs: dict) -> dict:
    composite_funcs_decl: dict = {}
    for func in funcs_info.keys():
        if func not in impl_funcs and op_configs.get(func, {}).get('supportComposite'):
            composite_funcs_decl[func] = funcs_decl_raw[func]
    return composite_funcs_decl


def gen_autogen_operators(dirs: dict, device: str,
                          adaptor_fm: FileManager) -> None:
    config_file_path = os.path.join(
        dirs.get('config_path'), 'convert_config.yaml')
    try:
        with open(config_file_path, 'r') as f:
            configs = yaml.safe_load(f)
    except Exception as e:
        print(e)
        return

    # get the implemented functions
    impl_func_dir = os.path.dirname(config_file_path)
    impl_func_dir = os.path.join(impl_func_dir, "functions")
    impl_funcs = obtain_impl_func(impl_func_dir).keys()

    # generate func information and declarations by scanning functions.h
    funcs_info, funcs_decl_raw = get_functions_support(dirs.get('source'))

    # get config information
    op_configs = analysis_configs(configs, funcs_info)

    # generate adaptor implementation codes
    adaptors_code = autogen_op_adaptor(
        op_configs, device, funcs_info, impl_funcs)

    # get the function declarations
    funcs_decl = get_impl_funcs_declaration(
        funcs_decl_raw, funcs_info, impl_funcs)
    composite_funcs_decl = get_composite_funcs_declaration(
        funcs_decl_raw, funcs_info, impl_funcs, op_configs)

    adaptor_fm.write('diopi_adaptor.cpp',
                     OT.operators_template,
                     dict(adaptors=adaptors_code,
                          cast_strategy=autogen_cast_strategy()))
    adaptor_fm.write('impl_functions.hpp',
                     OT.impl_declaration_template,
                     dict(device=device, impl_declaration=list(funcs_decl.values()), composite_funcs_decl=list(composite_funcs_decl.values())))


def declare_outputs(adaptor_fm: FileManager) -> None:
    adaptor_fm.will_write('diopi_adaptor.cpp')
    adaptor_fm.will_write('impl_functions.hpp')


def gen_all_codes() -> None:
    dirs, device = prepare()
    adaptor_fm = FileManager(dirs.get('output_dir', '.'))
    declare_outputs(adaptor_fm)
    gen_autogen_operators(dirs, device, adaptor_fm)
    adaptor_fm.check_all_files_written()


if __name__ == '__main__':
    gen_all_codes()
