import argparse
import os
import re
import yaml

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
    'float64': 'diopiDtype_t::diopi_dtype_float64',
    'float32': 'diopiDtype_t::diopi_dtype_float32',
    'float16': 'diopiDtype_t::diopi_dtype_float16',
    'bool': 'diopiDtype_t::diopi_dtype_bool',
}

str_to_diopi_format = {
    'NCHW': 'diopiMemoryFormat_t::Contiguous',
    'NHWC': 'diopiMemoryFormat_t::ChannelsLast'
}


default_cast_dtype = {
    'int64': 'int32',
    'float64': 'float32'
}

cast_strategy = {
    'NoCast'  : {},
    'Default' : {
        'int64': 'int32',
        'float64' : 'float32',
        'bool' : 'int32'
    },
    
    'CastFloatOnly' : {
        'int64': 'int32'
    },
    
    'LogicOp' : {
        'int64': 'int32',
        'float64' : 'int32'
    }
}


def prepare():
    parser = argparse.ArgumentParser(
        description='Generate parrots source files')
    parser.add_argument(
        '-s',
        '--diopi_dir',
        help='path of dependence used to generate code',
        default='../')
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output a list of source files into the given directory',
        default='./include')
    parser.add_argument(
        '-c',
        '--config_device',
        help='name of file which contains configs of device',
        default='camb')


    options = parser.parse_args()
    source = os.path.join(options.diopi_dir, 'DIOPI-PROTO/include/diopi')
    config_path = os.path.join(options.diopi_dir, 'DIOPI-IMPL/', options.config_device)
    def create_if_not_exist(name):
        if not os.path.exists(name):
            os.makedirs(name)

    create_if_not_exist(options.output_dir)
    dirs = dict(source=source,
                output_dir=options.output_dir,
                config_path=config_path)
    return dirs

def get_func_info(content):
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
                ins_v[tensor_name] = args[i+1].split(' ')[1]
    return ins, outs, args, ins_v


def get_functions_support(source_dir):
    with open(os.path.join(source_dir, 'functions.h'), 'r')as f:
        content = f.readlines()
    funcs_info = {}
    func_dtypes = []
    param_dtypes = {}
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
            if func_name not in funcs_info.keys():
                funcs_info[func_name] = {}
            funcs_info[func_name]['call_args'] = args
            if (param_dtypes == {} and func_dtypes == [] and sa_func == None) or ins == None:
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
                    funcs_info[func]['outs']['grad_' + i] = from_func_info['ins'][i]
            for out in from_func_info['outs']:
                if out in funcs_info[func]['ins']:
                    funcs_info[func]['ins'][out] = from_func_info['outs'][out]
                elif out in funcs_info[func]['outs']:
                    funcs_info[func]['outs'][out] = from_func_info['outs'][out]
                if 'grad_' + out in funcs_info[func]['ins']:
                    funcs_info[func]['ins']['grad_' + out] = from_func_info['outs'][out]
    return funcs_info

def deal_dtype(op_name, dtype_config, func_infos, tensor_name = None):
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
            from_dtypes = [d]
            to_dtype = default_cast_dtype[d]
        else: 
            from_dtypes = r.group(1).replace(' ', '').split(',')
            to_dtype = r.group(2)
        for f in from_dtypes:
            assert (op_name == 'Common' or not tensor_name) or \
                    (tensor_name in func_infos[op_name]['ins'].keys() and f in func_infos[op_name]['ins'][tensor_name]) or \
                    (tensor_name in func_infos[op_name]['outs'].keys())
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
    

def analysis_configs(config, funcs_info):
    common_cast = ''
    common_layout = []
    common_contiguous = False
    op_dict = {}
    for info in config:
        [op_name], [op_cfg] = info.keys(), info.values()
        if op_name == 'common_config':
            if 'dtype' in op_cfg.keys():
                common_cast = deal_dtype('Common', op_cfg['dtype'], funcs_info)
            if 'layout' in op_cfg.keys():
                common_layout = [op_cfg['layout']] if isinstance(op_cfg['layout'], str) else op_cfg['layout']
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
                    assert tensor in funcs_info[op_name]['ins'].keys() or  tensor in funcs_info[op_name]['outs'].keys()
                    tensor_cast = deal_dtype(op_name, op_cfg['tensor_dtype'][tensor], funcs_info, tensor)
                    op_tensor[tensor] = {}
                    op_tensor[tensor]['cast'] = tensor_cast 
            if 'layout' in op_cfg.keys():
                layouts = op_cfg['layout'].replace(' ', '').split(',')
                for layout in layouts:
                    if layout == 'NHWC' or layout == 'NCHW':
                        op_layouts.append(layout)
                    else:
                        r = re.match(r'(.*)\((.*)\)', layout)
                        tensor_name = r.group(1)
                        tensor_layout = r.group(2)
                        if tensor_name not in op_tensor.keys():
                            op_tensor[tensor] = {}
                            op_tensor[tensor]['layout'] = tensor_layout
                op_dict[op_name]['layout'] = op_layouts
            if 'contiguous' in op_cfg.keys():
                contiguous_tensor = op_cfg['contiguous'].replace(' ', '').split(',')
                for tensor_name in contiguous_tensor:
                    if tensor_name not in op_tensor.keys():
                        op_tensor[tensor] = {}
                    op_tensor[tensor]['contiguous'] = True
            for tensor in list(funcs_info[op_name]['ins'].keys())+list(funcs_info[op_name]['outs'].keys()):
                if tensor not in op_tensor.keys():
                    op_tensor[tensor] = {}
                if 'cast' not in op_tensor[tensor]:
                    op_tensor[tensor]['cast'] = op_cast if op_cast else common_cast
                if 'contiguous' not in op_tensor[tensor]:
                    op_tensor[tensor]['contiguous'] = True if common_contiguous else False
                if 'layout' not in op_tensor[tensor]:
                    op_tensor[tensor]['layout'] = op_layouts if len(op_layouts) else common_layout
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
        cast_code.append(OT.cast_strategy_template.substitute(env=dict(cast_name=strategy, cases=cases)))
    return cast_code


def memory_format_to_str(memory_format):
    default_format = ['NHWC', 'NCHW']
    is_default = [format in default_format for format in memory_format] and len(memory_format) == len(default_format)
    if len(memory_format) == 0 or is_default:
        return ''
    
    formats = []
    for format in memory_format:
        formats.append(str_to_diopi_format[format])
    return ', std::vector<diopiMemoryFormat_t>{' + ','.join(formats) + '}'


def autogen_op_adaptor(op_configs, func_infos):
    adaptors_code = []
    cast = op_configs['Common']['cast'] if 'Common' in op_configs.keys() else ''
    contiguous = op_configs['Common']['contiguous'] if 'Common' in op_configs.keys() else []
    layout = op_configs['Common']['layout'] if 'Common' in op_configs.keys() else []
    for func in func_infos:
        op_name = func.lstrip('diopi')
        if (func not in op_configs.keys() and 'Common' not in op_configs.keys()) or len(list(func_infos[func].keys())) == 1:
            call_args = [arg.split(' ')[-1] for arg in func_infos[func]['call_args']] 
            adaptors_code.append(OT.adaptor_template.substitute(env=dict(op_name=op_name, attrs=func_infos[func]['call_args'],
                             new_input='', cast_input='', cast_output='', call_func=func+'('+', '.join(call_args)+');')))
        else:
            op_config = op_configs[func] if func in op_configs.keys() else None
            new_ins = []
            cast_ins = []
            cast_outs = []
            new_input = []
            for tensor in list(func_infos[func]['ins'].keys()) + list(func_infos[func]['outs'].keys()):
                tensor_info = op_config['tensor'][tensor] if op_config else None
                contiguous_str = ', true' if (tensor_info and 'contiguous' in tensor_info.keys() \
                                and tensor_info['contiguous']) or contiguous else ''
                cast_method = tensor_info['cast'] if tensor_info else cast
                memory_format = tensor_info['layout'] if tensor_info else layout
                format_str = memory_format_to_str(memory_format)
                ins = func_infos[func]['ins']
                if tensor in ins:
                    if 'ins_vector' in func_infos[func].keys() and tensor in func_infos[func]['ins_vector'].keys():
                        new_ins_vector_template = CodeTemplate("""\
std::vector<diopiConstTensorHandle_t> ${newinput}(${num}, diopiConstTensorHandle_t());
for (int i = 0; i < ${num}; ++i) {
    castImpl<diopiConstTensorHandle_t${cast}${contiguous}>(ctx, ${input}[i], &${newinput}[i]${memory_format});
}
""")
                        new_input.append(new_ins_vector_template.substitute(env=dict(input=tensor, newinput='new'+tensor.capitalize(), num=func_infos[func]['ins_vector'][tensor],
                                                                                     cast=', ' + cast_method if cast_method else '', memory_format=format_str if format_str else '', contiguous=contiguous_str)))
                    else:
                        new_in = 'new' + tensor.capitalize()
                        new_ins.append(new_in)
                        cast_impl = 'castImpl<diopiConstTensorHandle_t{cast}{contiguous}>(ctx, {tensor}, &{new_tensor}{memory_format});'.format(
                                    cast=', ' + cast_method if cast_method else '', memory_format=format_str if format_str else '', tensor=tensor, new_tensor=new_in, contiguous=contiguous_str)
                        cast_ins.append(cast_impl)
                outs = func_infos[func]['outs']
                if tensor in outs:
                    cast_impl = 'auto {tensor}Wrapper = DiopiTensorWrapper<{cast}{contiguous}>(ctx, {tensor}{memory_format});'.format(
                                cast=cast_method, memory_format=format_str if format_str else '', tensor=tensor, contiguous=contiguous_str)
                    cast_outs.append(cast_impl)
            new_input.append('diopiConstTensorHandle_t ' + ','.join(new_ins) + ';') if len(new_ins) else ''
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
            
            adaptors_code.append(OT.adaptor_template.substitute(env=dict(op_name=op_name, attrs=', '.join(func_infos[func]['call_args']),
                                new_input=new_input, cast_input=cast_ins, cast_output=cast_outs, call_func=func+'('+', '.join(call_args)+');')))
    return adaptors_code

def gen_autogen_operators(dirs, adaptor_fm):
    config_file_path = os.path.join(dirs.get('config_path'), 'convert_config.yaml')
    try:
        with open(config_file_path, 'r') as f:
            configs = yaml.safe_load(f)
    except Exception as e:
        print(e)
        return
    
    funcs_info = get_functions_support(dirs.get('source'))
    op_configs = analysis_configs(configs, funcs_info)
    adaptors_code = autogen_op_adaptor(op_configs, funcs_info)
    casts_code = autogen_cast_strategy()

    adaptor_fm.write('diopi_adaptors.hpp',
                 OT.operators_template,
                 dict(adaptors=adaptors_code, cast_strategy=casts_code))


def declare_outputs(adaptor_fm):
    adaptor_fm.will_write('diopi_adaptors.hpp')


def gen_all_codes():
    dirs = prepare()
    adaptor_fm = FileManager(dirs.get('output_dir', '.'))
    declare_outputs(adaptor_fm)
    gen_autogen_operators(dirs, adaptor_fm)
    adaptor_fm.check_all_files_written()


if __name__ == '__main__':
    gen_all_codes()
