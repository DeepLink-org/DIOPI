# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-
import argparse
import os
import re
import yaml
import copy

from op_template import OpTemplate as OT
from filemanager import FileManager

tensor_ptr = ['diopiTensorHandle_t*', 'diopiConstTensorHandle_t*']

type_convert_dict = {
    'int64_t*': 'void*',
    'double*': 'void*'
}

can_be_none = ['const int64_t*', 'const double*', 'const bool*']


def prepare():
    parser = argparse.ArgumentParser(
        description='Generate parrots source files')
    parser.add_argument(
        '-d',
        '--device',
        help='name of device',
        default='torch'
        '-')

    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    options = parser.parse_args()
    source_dir = os.path.join(_cur_dir, '../proto/include/diopi/')
    output_dir = os.path.join(_cur_dir, '../csrc')
    diopilib_dir = os.path.join(_cur_dir, '../../python/diopilib/')
    device = options.device
    options = dict(source_dir=source_dir,
                   output_dir=output_dir,
                   diopilib_dir=diopilib_dir,
                   device=device)

    return options


def get_func_info(content):
    args = []
    attr_types = []
    paras_can_be_none = []
    ins_vector, outs_vector = {}, {}
    out_ptr = []
    type_change = False
    row = content.replace('\n', '').replace('(', ',').replace(')', '')
    arg_define = row.split(',')
    arg_index = 0
    for index, arg in enumerate(arg_define):
        arg = arg.strip(' ')
        temp = arg.split(' ')
        if arg != '' and temp[0] != 'diopiError_t':
            arg = temp[-1]
            arg_type = temp[0] + ' ' + temp[1] if len(temp) == 3 else temp[0]
            if arg_type in type_convert_dict:
                type_change = True
                arg_type = type_convert_dict[arg_type]
                arg = 'reinterpret_cast<' + temp[0] + '>(' + arg.split(' ')[-1] + ')'
            if arg_type in tensor_ptr:
                for i in range(index + 1, len(arg_define)):
                    next_arg = arg_define[i].strip(' ').split(' ')
                    if next_arg == 3:
                        assert arg_type == 'diopiTensorHandle_t*'
                        type_change = True
                        out_ptr.append(arg_index)
                        arg_type = 'PtrWrapper<diopiTensor>'
                        break
                    elif next_arg[0] == 'int64_t':
                        type_change = True
                        if arg_type == 'diopiTensorHandle_t*':
                            outs_vector[arg_index] = next_arg[1]
                        else:
                            ins_vector[arg_index] = next_arg[1]
                        arg_type = 'py::list&'
                        break
                    elif next_arg[0] in tensor_ptr:
                        continue
                    else:
                        type_change = True
                        assert arg_type == 'diopiTensorHandle_t*'
                        out_ptr.append(arg_index)
                        arg_type = 'PtrWrapper<diopiTensor>'
                        break
                if index == len(arg_define) - 1 and arg_type == 'diopiTensorHandle_t*':
                    type_change = True
                    out_ptr.append(arg_index)
                    arg_type = 'PtrWrapper<diopiTensor>'
            args.append(arg)
            attr_types.append(arg_type)
            if arg_type in can_be_none:
                paras_can_be_none.append(len(args) - 1)
            arg_index += 1
    return type_change, args, attr_types, paras_can_be_none, ins_vector, outs_vector, out_ptr


def gen_functions(options, functions_fm):
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    head_files = ['functions.h', 'functions_ext.h']
    exports = []
    for head_file in head_files:
        with open(os.path.join(_cur_dir, options.get('source_dir'), head_file), 'r', encoding='utf8') as f:
            content = f.readlines()
        device = options.get('device')
        if device == 'ascend':
            ft = OT.function_ascend_template
        else:
            ft = OT.function_template
        for idx, row in enumerate(content):
            if row.startswith("DIOPI_API"):
                row = row[10:]
                temp_content = ''
                idx1 = row.find("(")
                idx0 = row.rfind(" ", 0, idx1)
                func_name = row[idx0 + 1: idx1]
                if func_name == 'diopiCastDtype' or func_name == 'diopiCopyInp':
                    exports.append('m.def("{func_name}", {func_name});'.format(func_name=func_name))
                    continue
                temp_content += row.replace(';', '')
                idx2 = row.find(")")
                while idx2 == -1:
                    row1 = content[idx + 1]
                    idx2 = row1.find(")")
                    temp_content += row1.replace(';', '')
                    idx += 1
                type_change, args, attr_types, paras_none, ins_vector, outs_vector, out_ptr = get_func_info(temp_content)
                call_args = copy.deepcopy(args)
                type_change = True
                if type_change:
                    convert, out_copy = '', ''
                    for param_type in type_convert_dict:
                        temp_content = temp_content.replace(param_type, type_convert_dict[param_type])
                    attrs = []
                    for index in range(len(attr_types)):
                        if 'reinterpret_cast' in call_args[index]:
                            attrs.append(attr_types[index] + ' ' + call_args[index].split('(')[1].rstrip(')'))
                        else:
                            attrs.append(attr_types[index] + ' ' + call_args[index])
                    for vector in ins_vector:
                        convert += OT.vector_template.substitute(env=dict(param=call_args[vector], param_num=ins_vector[vector],
                                                                 param_type=attr_types[vector], handle_type='diopiConstTensorHandle_t'))
                        call_args[vector] = call_args[vector] + 'DIOPI'
                    for vector in outs_vector:
                        convert += OT.vector_template.substitute(env=dict(param=call_args[vector], param_num=outs_vector[vector],
                                                                 param_type=attr_types[vector], handle_type='diopiTensorHandle_t'))
                        call_args[vector] = call_args[vector] + 'DIOPI'
                    for out in out_ptr:
                        convert += "diopiTensorHandle_t {param}Handle = nullptr;\n".format(param=call_args[out])
                        out_copy += "if ({param}.get() != nullptr)\n \
        *{param} = *{param}Handle;\n".format(param=call_args[out])
                        call_args[out] = '&' + call_args[out] + 'Handle'
                    call_func = func_name + '(' + ', '.join(call_args) + ')'
                    exports.append(ft.substitute(env=dict(func_name=func_name, attrs=', '.join(attrs), convert=convert,
                                                          out_copy=out_copy, call_func=call_func)))
                else:
                    exports.append('m.def("{func_name}", {func_name});'.format(func_name=func_name))
                if len(paras_none):
                    arg_def = [attr_types[index] + ' ' + args[index] for index in range(len(args)) if index not in paras_none]
                    keep_args = []
                    for index, arg in enumerate(call_args):
                        keep_args.append(arg if index not in paras_none else 'nullptr')
                    call_func = func_name + '(' + ', '.join(keep_args) + ')'
                    if type_change:
                        exports.append(ft.substitute(env=dict(func_name=func_name, attrs=', '.join(arg_def), convert=convert,
                                                              out_copy=out_copy, call_func=call_func)))
                    else:
                        exports.append(ft.substitute(env=dict(func_name=func_name, attrs=', '.join(arg_def), convert='',
                                                              out_copy='', call_func=call_func)))

    functions_fm.write("export_functions.cpp", OT.operators_template, env=dict(export_functions=exports))


def declare_outputs(adaptor_fm):
    adaptor_fm.will_write('export_functions.cpp')


def lib_init(diopilib_dir):
    from lib_init_template import diopilib_init_tmp
    with open(diopilib_dir + '__init__.py', 'w') as f:
        f.write(diopilib_init_tmp)


def gen_all_codes():
    dirs = prepare()
    lib_init(dirs.get('diopilib_dir', '.'))
    functions_fm = FileManager(dirs.get('output_dir', '.'))
    declare_outputs(functions_fm)
    gen_functions(dirs, functions_fm)
    functions_fm.check_all_files_written()


if __name__ == '__main__':
    gen_all_codes()
