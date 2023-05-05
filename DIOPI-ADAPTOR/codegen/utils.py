import os
import re
from typing import Dict
import yaml
import copy

from .op_template import OpTemplate as OT
from code_template import CodeTemplate


rules = dict()

def set_rules(source_dir):
    global rules
    with open(os.path.join(source_dir, 'yamls', 'transfer_rules.yaml'), 'r') as f:
        rules = yaml.safe_load(f)


def get_comment(contents, paras, block=True):
    comment = get_op_comment(paras, block)
    if comment:
        contents.append(comment)


def get_op_comment(op_cfg, block=True):
    comment = op_cfg.get('comment')
    if comment:
        if block:
            contents = ('''
//===============================================
//
//   {}
//
//===============================================

'''.format(comment))
        else:
            contents = '// {}'.format(comment)
    else:
        contents = ''
    return contents


def get_namespace(namespace_str):
    namespace = namespace_str.split('::')
    namespace_start = '\n'.join(['namespace {} {{'.format(n)
                                 for n in namespace])
    namespace_end = '\n'.join(['}}  // namespace {}'.format(n)
                               for n in namespace])
    return namespace_start, namespace_end


def get_appendix_include(include):
    if isinstance(include, str):
        include = [include]
    include = ['#include <{}>'.format(i) for i in include]
    return include


def get_name_and_args(*args, aten=False):
    assert 2 <= len(args) <= 3
    if isinstance(args[0], int):
        if (args[0] > 1):
            names = ['{name}{idx}'.format(name=args[2], idx=i + 1)
                     for i in range(args[0])]
        else:
            names = [args[2] for i in range(args[0])]
    else:
        names = args[0]

    args = ['{} {}'.format(args[1], name) for name in names]

    if aten:
        names_aten = ['aten_{}'.format(name) for name in names]
        return names, args, names_aten
    else:
        return names, args, None


def analysis_attr(attrs, split=';'):
    attrs += split
    status = 0
    temp = ''
    num_brackets = 0
    num_angle_brackets = 0
    num_curly_brackets = 0
    aten_type = None
    base_type = None
    default_value = None
    result = []

    def isunalnum(c):
        return c.isalnum() or c in ('_', ':')

    if attrs is None or attrs == '':
        return []

    for c in attrs:
        # start
        if status == 0:
            if isunalnum(c):
                temp += c
                status = 1
            elif c in (' ', split):
                status = 2
            else:
                status = None
        # get base type
        elif status == 1:
            if isunalnum(c):
                temp += c
            elif c == '<':
                temp += c
                num_angle_brackets += 1
                status = 3
            else:
                aten_type = temp
                name = temp
                temp = ''
                if c == ' ':
                    status = 4
                elif c == '(':
                    num_brackets = 1
                    status = 5
                elif c == '=':
                    aten_type = None
                    status = 9
                else:
                    result.append((None, None, name, None))
                    status = 2
        # before base type get(success status)
        elif status == 2:
            base_type = None
            default_value = None
            if isunalnum(c):
                temp += c
                status = 1
            elif c not in (' ', split):
                status = None
        # get template args
        elif status == 3:
            temp += c
            if c == '<':
                num_angle_brackets += 1
            elif c == '>':
                num_angle_brackets -= 1
                if num_angle_brackets == 0:
                    aten_type = temp
                    temp = ''
                    status = 4
        # after base type get
        elif status == 4:
            if isunalnum(c):
                temp += c
                status = 7
            elif c == '(':
                num_brackets = 1
                status = 5
            elif c == '=':
                status = 9
            elif c != ' ':
                status = None
        # get target type
        elif status == 5:
            if c == ')':
                num_brackets -= 1
                if num_brackets == 0:
                    base_type = temp
                    temp = ''
                    status = 6
                else:
                    temp += c
            elif c != split:
                temp += c
                if c == '(':
                    num_brackets += 1
            else:
                status = None
        # after target type get
        elif status == 6:
            if isunalnum(c):
                temp += c
                status = 7
            elif c != ' ':
                status = None
        # get variable name
        elif status == 7:
            if c in (' ', '=', split):
                name = temp
                temp = ''
                if c == ' ':
                    status = 8
                elif c == '=':
                    status = 9
                else:
                    result.append((aten_type, base_type, name, None))
                    status = 2
            elif isunalnum(c):
                temp += c
            else:
                status = None
        # after name get
        elif status == 8:
            if c == split:
                result.append((aten_type, base_type, name, None))
                status = 2
            if c == '=':
                status = 9
            elif c != ' ':
                status = None
        # before value get
        elif status == 9:
            if c == '"':
                temp += c
                status = 11
            elif c == '{':
                temp += c
                num_curly_brackets += 1
                status = 12
            elif c == split:
                status = None
            elif c != ' ':
                temp += c
                status = 10
        # get default value
        elif status == 10:
            if c == split:
                default_value = temp.strip()
                temp = ''
                result.append((aten_type, base_type, name, default_value))
                status = 2
            elif c == '"':
                temp += c
                status = 11
            elif c == '{':
                temp += c
                num_curly_brackets += 1
                status = 12
            else:
                temp += c
        # get string in default value
        elif status == 11:
            temp += c
            if c == '"':
                status = 10
        # get vector or array in default value
        elif status == 12:
            temp += c
            if c == '{':
                num_curly_brackets += 1
            elif c == '}':
                num_curly_brackets -= 1
                status = 10 if num_curly_brackets == 0 else 12
        elif status is None:
            break
    assert status == 2, ('Given string must fallow rule: '
                         '(base_type)[(target_type)] name[ = value;]')
    return result


def get_arg_code_from_attrs(attrs):
    result = analysis_attr(attrs)
    assert len(result) > 0
    codes = []
    for attr in result:
        codes.append('{} {};'.format(attr[0], attr[2]))
    gets = 'SSAttrs(attr_)'
    for attr in result:
        if attr[3]:
            gets += '.get("{}", {}, {})'.format(attr[2], attr[2], attr[3])
        else:
            gets += '.get("{}", {})'.format(attr[2], attr[2])
    gets += ';'
    codes.append(gets)
    return codes, result


def get_transfer_from_attrs(result, paras_list, scalars=None, constant=None):
    global rules
    codes = []
    attr_names = []
    for attr in result:
        if attr[1] is not None:
            rule = rules.get(attr[1], rules['default'])
            if isinstance(rule, dict):
                rule = rule.get(attr[0], rule.get('other', rules['default']))
            # not generate ATen variable of attr not in para_list
            if attr[2] in paras_list:
                codes.append(rule.format(target=attr[1], name=attr[2]))
                attr_names.append(attr[2] + 'ATen')
        else:
            attr_names.append(attr[2])
    if scalars:
        for scalar in scalars:
            # not generate ATen variable of scalar not in para_list
            if scalar[2] in paras_list:
                codes.append(OT.scalar_trans_template.substitute(
                    env=dict(name=scalar[2])))
                attr_names.append(scalar[2] + 'ATen')
    if constant is not None:
        const_cfg = analysis_attr(constant, ',')

        scalar_trans = 'at::Scalar {name}ATen({value});'
        for const in const_cfg:
            if const[2] in paras_list:
                codes.append(scalar_trans.format(name=const[2], value=const[3]))
                attr_names = [const[2] + 'ATen']
            else:
                continue
        # attr_names = [constant + 'ATen']
    return codes, attr_names


def get_diopi_attrs(attrs, ins, outs, para_used, scalars=None, constant=None, ins_name_vector=None, outs_name_vector=None):
    diopi_rules = {
        'auto': '{type} {name}DIOPI = {value};',
        'diopiSize_t': '{type} {name}DIOPI = parrots::diopi::toDiopiSize({value});',
        'diopiScalar_t': '{type} {name}DIOPI = parrots::diopi::toDiopiScalar({value});',
        'diopiReduction_t': '{type} {name}DIOPI = parrots::diopi::toDiopiReduction({value});',
        'diopiDtype_t': '{type} {name}DIOPI = parrots::diopi::toDiopiDtype({value});',
        'diopiRoundMode_t': '{type} {name}DIOPI = parrots::diopi::toDiopiRoundMode({value});',
        'scalars': {
            'int64_t': '{type} {name}DIOPI = parrots::diopi::scalarToValue<{type}>({value});',
            'double': '{type} {name}DIOPI = parrots::diopi::scalarToValue<{type}>({value});',
            'float': '{type} {name}DIOPI = parrots::diopi::scalarToValue<{type}>({value});',
            'bool': '{type} {name}DIOPI = parrots::diopi::scalarToValue<{type}>({value});',
        }
    }

    used_para_names = para_used.keys()
    codes = []
    call_paras = []

    def gen_diopi_attr_codes(name):
        diopi_type = para_used.get(name)
        if diopi_type.endswith('*'):
            call_paras.append('&' + name + 'DIOPI')
            diopi_type = diopi_type[:-1]
        else:
            call_paras.append(name + 'DIOPI')
        return diopi_type
    # attr : [parrots_type, aten_type, name]
    attr_names = [attr[2] for attr in attrs] if attrs is not None else []
    scalar_names = [s[2] for s in scalars] if scalars else []
    consts = {}
    if constant is not None:
        const_cfg = analysis_attr(constant, ',')
        for c in const_cfg:
            consts[c[2]] = c[3]

    out_need_update = []
    for p in used_para_names:
        if p == 'nullptr':
            call_paras.append('nullptr')
        elif p == 'ctx':
            continue
        elif p in ins or p.lstrip('&') in outs:
            if p.startswith('&') and p.lstrip('&') in outs:
                out_need_update.append(p.lstrip('&'))
            call_paras.append(p + 'DIOPI')
            if p in ins_name_vector or p.lstrip('&') in outs_name_vector:
                call_paras.append(p + '.size()')
        elif p in attr_names:
            diopi_type = gen_diopi_attr_codes(p)
            rule = diopi_rules.get(diopi_type, diopi_rules['auto'])
            codes.append(rule.format(type=diopi_type, name=p, value=p))
        elif p in scalar_names:
            diopi_type = gen_diopi_attr_codes(p)
            rule = diopi_rules['scalars'].get(diopi_type, diopi_rules.get(diopi_type, diopi_rules['auto']))
            codes.append(rule.format(type=diopi_type, name=p, value=p))
        elif p in consts.keys():
            diopi_type = gen_diopi_attr_codes(p)
            rule = diopi_rules.get(diopi_type, diopi_rules['auto'])
            codes.append(rule.format(type=diopi_type, name=p, value=consts[p]))
        elif p.endswith('Dtype'):
            diopi_dtype = para_used.get(p)
            relu = diopi_rules.get(diopi_dtype)
            codes.append(relu.format(type=diopi_dtype, name=p, value=p[:-5]))
            call_paras.append(p + 'DIOPI')
        elif p.endswith('DiopiSize'):
            diopi_dtype = para_used.get(p)
            rule = diopi_rules.get(diopi_dtype)
            codes.append(rule.format(type=diopi_dtype, name=p, value=p.rstrip('DiopiSize')))
            call_paras.append(p + 'DIOPI')
            
    for out in out_need_update:
        para_used[out] = para_used['&' + out]
        del para_used['&' + out]
    return codes, call_paras, out_need_update


def get_tensor_name(ins):
    if ins == None or ins == 0:
        ins = ''
    return [c.strip() for c in re.split(r'[ ,;]', ins) if c]

def check_ins_outs(ins_name, outs_name):
    in_has_optional = False
    out_has_optional = False
    for name in ins_name:
        if in_has_optional:
            assert name.endswith('?'), ('optional ins or outs needs to be configured at the end')
        if name.endswith('?'):
            in_has_optional = True
    for name in outs_name:
        if out_has_optional:
            assert name.endswith('?'), ('optional ins or outs needs to be configured at the end')
        if name.endswith('?'):
            out_has_optional = True

def get_darrays(ins, type, spec=None):
    ins_darray = []
    prefix = 'const ' if type == 'ins' or type == 'inSpecs' else ''
    vector_type = 'std::vector'
    darray_template, darray_template_vector = '', ''
    if spec == 'infer':
        def_type = 'DArraySpec'
        vector_type = 'vector_t'
    else:
        def_type = 'DArrayLite'
        vector_type = 'std::vector'
    darray_template = prefix + '{def_type} &{in_name} = {type}[{in_offset}];'
    darray_template_vector = prefix + '{vector_type}<{def_type}> {in_name}({type}.begin(){start_size}, {type}.end(){end_size});'
    if type == 'inSpecs' or type == 'ins':
        darray_template_optinal = prefix + '{def_type} &{in_name} = {type}.size() == {ins_size} ? {type}[{in_offset}] : {def_type}();'
    else:
        darray_template_optinal = '{def_type} temp;\n{def_type} &{in_name} = {type}.size() == {ins_size} ? {type}[{in_offset}] : temp;'

    start_size, end_size = 0, 0
    is_vector = False
    vector_name = ""
    for in_offset, in_name in enumerate(ins):
        if not '[]' in in_name:
            start_size += 1
            if in_name.endswith('?'):
                ins_darray.append(darray_template_optinal.format(in_name=in_name.strip('?'),
                                            in_offset=in_offset,
                                            type=type,
                                            ins_size=len(ins),
                                            def_type=def_type,
                                            vector_type=vector_type))
            else:
                ins_darray.append(darray_template.format(in_name=in_name,
                                                         in_offset=in_offset,
                                                         type=type,
                                                         def_type=def_type,
                                                         vector_type=vector_type))
        else:
            is_vector = True
            vector_name = in_name[:-2].rstrip('?')
            break
    if is_vector is False:
        return ins_darray
    for in_offset, in_name in enumerate(reversed(ins)):
        if not '[]' in in_name:
            end_size += 1
            ins_darray.append(darray_template.format(in_name=in_name,
                                                     in_offset=f"{type}.size() - {in_offset + 1}",
                                                     type=type,
                                                     def_type=def_type,
                                                     vector_type=vector_type))
        else:
            break
    ins_darray.append(darray_template_vector.format(in_name=vector_name,
                                                    start_size=' + ' + str(start_size) if start_size != 0 else '',
                                                    end_size=' - ' + str(end_size) if end_size != 0 else '',
                                                    type=type,
                                                    def_type=def_type,
                                                    vector_type=vector_type))
    return ins_darray

def get_outs(outs_name, ins_name):
    out_attrs = []
    names = []
    if outs_name is None:
        return out_attrs, names
    outs = []
    brackets_num = 0
    single_out = ''
    for c in outs_name:
        if c == ' ':
            continue
        elif c == ',' or c == ';':
            if brackets_num == 0:
                outs.append(single_out)
                single_out = ''
                continue
        elif c == '(':
            brackets_num += 1
        elif c == ')':
            brackets_num -= 1
        single_out += c
    if single_out != '':
        outs.append(single_out)

    storage_options = {'shared' : 'StorageOption::Shared',
                      'pinned' : 'StorageOption::Pinned',
                      'none' : 'StorageOption::None'}
    make_options = {'nonelike' : 'noneLike',
                   'emptylike' : 'emptyLike',
                   'rawlike' : 'rawLike',
                   'none' : 'none',
                   'empty' : 'empty',
                   'fulllike' : 'fullLike',
                   'createreducecastout' : 'createReduceCastOut',
                   'createnotnone' : 'createNotNone'}
    for out in outs:
        if out == None:
            continue
        re.match(r'(.+ *)\((.+)\)', out)
        r = re.match(r'([a-zA-Z0-9_\?\[\]]*)(\([a-zA-Z0-9_:]*\))?( *= *([a-zA-Z0_9,:(){}\[\]]*)\((.*)\))?', out)
        storage_option = None
        shared_from = None
        make_out = None
        if r.group(2):
            storage_set = r.group(2).rstrip(')').lstrip('(')
            storage_set = storage_set.split(':')
            assert storage_set[0].lower() in storage_options.keys()
            assert len(storage_set) == 1 or storage_set[0].lower() == 'shared'
            storage_option = storage_options[storage_set[0].lower()]
            if storage_option == "StorageOption::Shared":
                shared_from = storage_set[1]
                assert shared_from in ins_name
        if r.group(4):
            make_type = r.group(4).lower()
            from_list = r.group(5).split(',')

            make_out = r.group(3).replace(' ', '').replace('=', '')
        out_attrs.append([r.group(1), storage_option, shared_from, make_out])
        names.append(r.group(1))

    return out_attrs, names

def _get_attrs(attrs, attr_form):
    if len(attrs) == 0:
        return []
    attr_template = '{base_type} __attribute__((unused)) {attr_name};'
    attrs_get = ('SSAttrs({name}.isNil() ? *parrots::SSElement::createMap() '
                 ': {name})'.format(name=attr_form))
    attrs_get_template = '.get("{attr_name}", {attr_name}, {default_value})'
    attrs_get_template_novalue = '.get("{attr_name}", {attr_name})'

    attrs_state = []
    for attr in attrs:
        attrs_state.append(attr_template.format(base_type=attr[0],
                                                attr_name=attr[2]))
        if attr[3] is None:
            attrs_get += attrs_get_template_novalue.format(attr_name=attr[2])
        else:
            attrs_get += attrs_get_template.format(attr_name=attr[2],
                                                   default_value=attr[3])

    attrs_state.append(attrs_get + ';')
    return attrs_state


def get_attrs_statement(attrs, scalars, attr_form):
    attr_state = _get_attrs(attrs, attr_form)
    attr_template = 'auto *p{name} = {attr_form}.getPtr(\"{name}\");\n'\
        'const SSElement::ScalarBase& __attribute__((unused)) {name} = p{name} ? p{name}->scalar() : '\
        'SSElement::createScalar({default_value})->scalar();'
    if scalars:
        for scalar in scalars:
            if scalar[3] is None:
                attr_state.append('const SSElement::ScalarBase& __attribute__((unused)) {} = {}[\"{}\"'
                    '].scalar();'.format(scalar[2], attr_form, scalar[2]))
            else:
                attr_state.append(attr_template.format(name=scalar[2],
                    attr_form=attr_form, default_value=scalar[3]))

    return attr_state


def get_dispatch(ins_name, outs_name, attrs_cfg, op_name,
                 dispatch, default_backend,  use_aten, use_diopi, use_native, direct_call, op_type,
                 scalars=[]):
    attrs = [attr[2].replace('backend', 'backend_s') for attr in attrs_cfg]
    if scalars:
        attrs += [s[2] for s in scalars]
    args = ', '.join([name.rstrip('[]').rstrip('?') for name in outs_name] +
                     [name.rstrip('[]').rstrip('?') for name in ins_name] + attrs)
    case_map = {'aten': 'ATen', 'native': 'Native', 'diopi': 'DIOPI'}
    case_code = []

    #diopi used native info
    native_infos = None
    native_call = None
    default_call = None
    diopi_call = None

    aten_tpl = CodeTemplate("""\
{
    ${op_name}(ctx, ${args});
}
""")
    native_tpl = CodeTemplate("""\
{
    PARROTS_DISPATCHOPF_BY_CTX(${name}, ${args});
}
""")
    native_pass_tpl = CodeTemplate("""\
{
    if (ctx.getProxy().arch() == Arch::X86) { 
        pass<Host>(${args});
    } else {
#if PARROTS_USE_DEVICE
        pass<Device>(${args});
#endif
    }
}    
""")
    diopi_tpl = CodeTemplate("""\
{
    if (ctx.getProxy().arch() == Arch::X86) { 
        ${host_call_code}
    } else {
#if PARROTS_USE_DEVICE
        ${diopi_code}
#endif
    }
}
""")
    comm_tpl = CodeTemplate("""\
{
    if (ctx.getProxy().arch() == Arch::X86) {
#ifdef PARROTS_USE_MPI
        ${name}<Host>(${args});
#endif
    } else {
#if defined(PARROTS_USE_NCCL) || defined(PARROTS_USE_RCCL) || defined(PARROTS_USE_CNCL) || defined(PARROTS_USE_HCCL)
        ${name}<Device>(${args});
#endif
    }
}
""")

    dispatch_call = ''
    case_automatic = 'case Backend::automatic:'
    case_cudnn = 'case Backend::cudnn:\n    PARROTS_LOG_WARN_ONCE("Parrots will not support para \'backend\'=\'cudnn\' after v0.18.0,"\n     "because we use backend cudnn by default, please remove this para.")'
    case_code.append(case_cudnn)
    for b in dispatch.keys():
        if (b == 'aten') and use_aten:
            if default_backend == 'aten' and (not use_diopi or 'diopi' not in dispatch.keys()):
                case_code.append(case_automatic)
            aten_interface = dispatch.get(b, '')
            r = re.match(r'(.+ *)\((.+)\)', aten_interface)
            if b in direct_call:
                dispatch_call = aten_interface.rstrip(';') + ';'
            else:
                dispatch_call = aten_tpl.substitute(env=dict(
                    op_name=op_name + case_map[b], args=args
                    ))
        elif (b == 'diopi') and use_diopi:
            case_code.append(case_automatic)
            host_call_code = ''
            if default_backend == 'native' or ('aten' not in dispatch.keys() or not use_aten):
                if use_native and ('native' in dispatch.keys()):
                    dispatch_call = dispatch.get('native', '').rstrip(';') + ';'
                    r = re.match(r'(.+ *)\((.+)\)', dispatch_call)
                    if r is not None:
                        if 'native' not in direct_call:
                            if r[1] == 'pass':
                                host_call_code = f'pass<Host>({r[2]});'
                            else:
                                host_call_code = f'PARROTS_CALLOPF({r[1]}, Host, {r[2]});'
                        else:
                            host_call_code = dispatch_call
                if host_call_code == '':
                    host_call_code = f'PARROTS_NOTSUPPORTED << "{op_name} don\'t have cpu(native) implement.";'
            else:
                aten_interface = dispatch.get('aten', '')
                r = re.match(r'(.+ *)\((.+)\)', aten_interface)
                if 'aten' in direct_call:
                    host_call_code = aten_interface.rstrip(';') + ';'
                else:
                    host_call_code = aten_tpl.substitute(env=dict(
                        op_name=op_name + case_map['aten'], args=args
                        ))
            diopi_dispatch = dispatch.get(b)
            if b in direct_call:
                diopi_code = diopi_dispatch
            else:
                diopi_call_name = op_name + case_map[b]
                diopi_code = '{name}(ctx, {args});'.format(name=diopi_call_name, args=args)
            dispatch_call = diopi_tpl.substitute(env=dict(
                host_call_code=host_call_code,
                diopi_code=diopi_code
            ))

            diopi_call = dispatch_call
        elif (b == 'native') and use_native:
            if (default_backend == 'aten' and (not use_aten or 'aten' not in dispatch.keys())) or \
               (default_backend == 'native' and (not use_diopi or 'diopi' not in dispatch.keys())):
               case_code.append(case_automatic)
            dispatch_call = dispatch.get(b, '').rstrip(';') + ';'
            r = re.match(r'(.+ *)\((.+)\)', dispatch_call)
            if r is not None:
                if b not in direct_call:
                    if r[1] == 'pass':
                        dispatch_call = native_pass_tpl.substitute(env=dict(args=r[2]))
                    else:
                        if op_type == 'comm':
                            dispatch_call = comm_tpl.substitute(env=dict(name=r[1], args=r[2]))
                        else:
                            dispatch_call = native_tpl.substitute(env=dict(name=r[1], args=r[2]))
                else:
                    dispatch_call = dispatch_call
                native_infos = r
                native_call = dispatch_call
        else:
            continue

        if b == default_backend:
            default_call = dispatch_call

        case_code.append(OT.dispatch_case_template.substitute(env=dict(
            op_name=op_name,
            dispatch_call=dispatch_call,
            backend=b
        )))
    # add Backend::automatic
    if use_diopi and 'diopi' in dispatch.keys():
        if native_call is None:
            native_call = f'PARROTS_NOTSUPPORTED << "{op_name} don\'t have cpu(native) implement.";'
        dispatch_call = OT.dispatch_automatic_diopi_tpl.substitute(env=dict(
            call_diopi=diopi_call,
            call_default=native_call,
        ))
    else:
        if default_call is None:
            dispatch_call = f'PARROTS_NOTSUPPORTED << "{op_name} don\'t have implement.";'
        else:
            dispatch_call = default_call

    if len(case_code) == 0:
        case_code.append(case_automatic)
    dispatch_code = OT.dispatch_template.substitute(env=dict(
        case=case_code,
    ))

    return dispatch_code, native_infos


def analysis_aten_func(interface):
    if interface.startswith('invokeATenFuncRet') or \
            interface.startswith('invokeATenFuncOut') or \
            interface.startswith('invokeATenFuncInp') or 'ATen(' in interface:
        r = re.match(r'(.+ *)\((.+)\)', interface)
        args = r.group(2).replace(' ', '').split(',')
        return r.group(0), r.group(1), args

    r = re.match(r'(.+) (.+ *)\((.+)\)', interface)
    if r:
        args = r.group(3).replace(' ', '').split(',')
        return r.group(1), r.group(2), args
    else:
        r = re.match(r'(.+ *)\((.+)\)', interface)
        args = r.group(2).replace(' ', '').split(',')
        return None, r.group(1), args


def analysis_diopi_func(interface):
    r = re.match(r'(.+ *)\((.+)\)', interface)
    args = r.group(2).split(', ')
    para_dict = dict()
    for arg in args:
        rst = re.match(r'(.+ )(.+)', arg)
        # long long a -> 'long long ','a'; bool* a -> 'bool* ', 'a'
        if rst is not None:
            name = rst.group(2)
            para_dict[name] = rst.group(1)[:-1]
        else:
            name = arg
            para_dict[name] = 'auto'
    return r.group(1), para_dict


def wrap_infer_func(inferfunc, ins_name, outs_name):
    r = re.match(r'(.+ *)\((.+)\)', inferfunc)
    args = r.group(2).replace(' ', '').split(',')
    ins_vector, outs_vector = False, False
    for in_name in ins_name:
        if in_name.endswith('[]'):
            ins_vector = True
            break
    for out_name in outs_name:
        if out_name.endswith('[]'):
            outs_vector = True
            break
    def transfer(name):
        if name.endswith('?'):
            name = name[: -1]
        if name in ins_name:
            if not ins_vector:
                name = 'inSpecs[{}]'.format(ins_name.index(name))
        elif name + '?' in ins_name:
            if not ins_vector:
                name = 'inSpecs.size() >= {in_size} ? inSpecs[{in_offset}] : DArraySpec()'.format(in_size=ins_name.index(name+'?')+1, in_offset=ins_name.index(name+'?'))
        elif name in outs_name:
            if not outs_vector:  
                name = 'outSpecs[{}]'.format(outs_name.index(name))
        elif name +'?' in outs_name:
            if not outs_vector:
                name = 'outSpecs.size() >= {out_size} ? outSpecs[{out_offset}] : DArraySpec()'.format(out_size=outs_name.index(name+'?')+1, out_offset=outs_name.index(name+'?'))
        elif name + '[]' in outs_name and len(outs_name) == 1:
            name = 'outSpecs'
        elif name + '[]' in ins_name and len(ins_name) == 1:
            name = 'inSpecs'
        return name
    calc = ''
    tensors = ''
    if ins_vector and len(ins_name) != 1:
        tensors += '\n'.join(get_darrays(ins_name, 'inSpecs', 'infer'))
        tensors += '\n'
    if outs_vector and len(outs_name) != 1:
        tensors += '\n'.join(get_darrays(outs_name, 'outSpecs', 'infer'))
        tensors += '\n'
    req = [transfer(n) for n in args if n.endswith('?')]
    calc += r.group(1) + '(' + ', '.join([transfer(n) for n in args]) + ');'
    return req, calc, tensors


def transfer(name, ins_name, outs_name, attrs_scalars, scalars, defaults, array_type = 'DArrayLite'):
    attr_names = [a[2] for a in attrs_scalars]
    scalar_names = [a[2] for a in scalars]
    vector_name = 'std::vector' if array_type == 'DArrayLite' else 'vector_t'
    if name[-2:] == '[]' and name in ins_name:
        return 'const '+ vector_name + '<' + array_type + '>& ' + name[:-2].rstrip('?')
    elif name[-2:] == '[]' and name in outs_name:
        return vector_name + '<' + array_type + '>& ' + name[:-2].rstrip('?')
    elif name == 'ctx':
        return 'ContextBase& ctx'
    elif name == 'attr_':
        return 'const SSElement& ' + name
    elif name in ins_name or name + '?' in ins_name:
        return 'const ' + array_type + '& ' + name.strip('?')
    elif name in outs_name or name + '?' in outs_name:
        return '' + array_type + '& ' + name.strip('?')
    elif name in attr_names:
        attr = attrs_scalars[attr_names.index(name)]
        attr_type = attr[0] if name not in scalar_names else ('const Scalar&' if array_type == 'DArrayLite' else 'Scalar')
        if attr[3]:
            defaults.append('{} {} = {}'.format(attr_type, attr[2], attr[3]))
        else:
            return attr_type + ' ' + attr[2]
    return None

def build_new_info(op_cfg, func, aten_func, diopi_func):
    op_cfg['funcs'] = None
    op_cfg['aten_funcs'] = None
    op_cfg['diopi_funcs'] = None

    aten = ['aten', 'aten*', 'aten(D)', 'aten(D)*']
    native = ['native', 'native*', 'native(D)', 'native(D)*']
    diopi = ['diopi', 'diopi*', 'diopi(D)', 'diopi(D)*']
    dispatch = op_cfg.get('dispatch', None)
    if dispatch:
        for d in dispatch.keys():
            if d in aten:
                replace_func = aten_func
            elif d in native:
                replace_func = func
            elif d in diopi:
                replace_func = diopi_func
            dispatch[d] = dispatch[d].replace('${name}', replace_func)

    if 'ret' in op_cfg:
        op_cfg['ret'] = op_cfg['ret'].replace('${name}', func)
    if 'interface' in op_cfg:
        for type in op_cfg['interface'].keys():
            if type.rstrip('(I)') == 'inplace' and 'inp${name}' in op_cfg['interface'][type]:
                op_cfg['interface'][type] = op_cfg['interface'][type].replace('${name}', func[0].upper() + func[1:])
            else:
                op_cfg['interface'][type] = op_cfg['interface'][type].replace('${name}', func)
    inherited = op_cfg.get('inherited', None)
    if inherited:
        name = func.replace('inp', '').replace('Out','')
        op_cfg['inherited'] = op_cfg['inherited'].replace('${name}', name[0].lower() + name[1:])
    return func, op_cfg


def unpack_funcs(funcs, op_cfg, op_dict):
    aten_funcs = op_cfg.get('aten_funcs', None)
    diopi_funcs = op_cfg.get('diopi_funcs', None)
    funcs = funcs.replace(' ', '').split(',')
    if aten_funcs is None:
        aten_funcs = ['...']
    else:
        aten_funcs = aten_funcs.replace(' ', '').split(',')
    if diopi_funcs is None:
        diopi_funcs = ['...']
    else:
        diopi_funcs = diopi_funcs.replace(' ', '').split(',')
    aten_space = len(funcs) - len(aten_funcs)
    diopi_space = len(funcs) - len(diopi_funcs)
    aten_index = 0
    diopi_index = 0
    for func in funcs:
        aten_func = aten_funcs[aten_index]
        if aten_func == '.':
            aten_func = func + '_out'
        elif aten_func == '...':
            aten_func = func + '_out'
            if aten_space > 0:
                aten_space = aten_space - 1
                aten_index = aten_index - 1
        diopi_func = diopi_funcs[diopi_index]
        if diopi_func == '.':
            diopi_func = func[0].upper() + func[1:]
        elif diopi_func == '...':
            diopi_func = func[0].upper() + func[1:]
            if diopi_space > 0:
                diopi_space = diopi_space - 1
                diopi_index = diopi_index - 1
        diopi_index = diopi_index + 1
        aten_index = aten_index + 1
        op_name, single_cfg = build_new_info(copy.deepcopy(op_cfg), func, aten_func, diopi_func)
        op_dict[op_name] = single_cfg


def preprocess_cfg(infos):
    op_dict = {}
    for info in infos:
        [op_name], [op_cfg] = info.keys(), info.values()
        funcs = op_cfg.get('funcs', None)
        if funcs is not None:
            unpack_funcs(funcs, op_cfg, op_dict)
        else:
            op_dict[op_name] = op_cfg
    return op_dict
        

def get_parent_cfg(op_dict, op_cfg, inherited, op_name):
    parrent = op_dict[inherited]
    for k in parrent.keys():
        if k in op_cfg.keys():
            continue
        if k == 'ret':
            op_cfg[k] = parrent[k]
            if isinstance(op_cfg[k], dict):
                for func in op_cfg[k].keys():
                    op_cfg[k][func] = op_cfg[k][func].replace(inherited, op_name)
            else:
                op_cfg[k] = op_cfg[k].replace(inherited, op_name)
        else:
            op_cfg[k] = parrent[k]


def check_args(op_name, op_cfg):
    key_args = ['ins', 'outs', 'constant', 'attr', 'scalars',
                'infer', 'inferfunc', 'check', 'comment', 'diopi_funcs',
                'dispatch', 'ret', 'sup_hybrid_arch', 'backendfunc', 'op_type',
                'funcs', 'aten_funcs', 'type_convert', 'strided', 'interface', 'inherited']
    args = {'infer': ['check', 'requirements', 'calculate', 'use_attr'],
            'dispatch': ['native', 'native*', 'native(D)', 'native(D)*',
                         'aten', 'aten*', 'aten(D)', 'aten(D)*',
                         'diopi', 'diopi(D)', 'diopi(D)*'],
            'ret': ['func', 'aten', 'share', 'interfaces', 'ctx'],
            'interface': ['ret', 'ret(I)', 'out', 'out(I)', 'inplace', 'inplace(I)']}
    for key in op_cfg.keys():
        assert key in key_args, (f"Invalid argument in {op_name}, "
                                  "key '{key}' not in {key_args}")
        tmp = args.get(key, None)
        if tmp is None:
            continue
        else:
            if isinstance(op_cfg.get(key), str):
                continue
            for k in op_cfg.get(key).keys():
                assert k in tmp, (f"Invalid argument in {op_name}, "
                                   "key '{k}' not in {tmp}")
                
def reorder_args(args):
    attrs_names = []
    attrs_args = []
    default = []
    for attr in args:
        if attr[3]:
            default.append(attr)
        else:
            attrs_args.append(attr)
            attrs_names.append(attr)
    attrs_args += default
    attrs_names = [a[2] for a in attrs_args]
    return attrs_args, attrs_names

def get_vector(names):
    vector = None
    for name in names:
        if name[-2:] == '[]':
            vector = name[:-2].rstrip('?')
    return vector

def check_ins_outs_nums(ins_name, outs_name, type = 'infer'):
    # check if the DArray number of ins and outs is correct
    checknums_template = CodeTemplate("""\
PARROTS_CHECKARGS(${check_in});
PARROTS_CHECKARGS(${check_out});
""")
    if type == 'infer':
        checkin = "inSpecs.size() {ins_check} {ins_num}"
        checkout = "outSpecs.size() {outs_check} {outs_num}"
    elif type == 'opclass':
        checkin = "ins.size() {ins_check} {ins_num}"
        checkout = "outs.size() {outs_check} {outs_num}"
        
    ins_check, outs_check = '==', '=='
    out_optional_nums = 0
    in_optional_nums = 0
    for in_name in ins_name:
        if in_name[-2:] == '[]':
            ins_check = '>='
        if '?' in in_name:
            in_optional_nums += 1
    for out_name in outs_name:
        if out_name[-2:] == '[]':
            outs_check = '>='
        if '?' in out_name:
            out_optional_nums += 1

    check_in = checkin.format(ins_check = ins_check, ins_num = len(ins_name))
    for i in range(in_optional_nums):
        check_in += ' || ' + checkin.format(ins_check = ins_check, ins_num = len(ins_name)-i-1)
    check_out = checkout.format(outs_check = outs_check, outs_num = len(outs_name))
    for i in range(out_optional_nums):
        check_out += ' || ' + checkout.format(outs_check = outs_check, outs_num = len(outs_name)-1)
    checknums = checknums_template.substitute(env=dict(
        check_in=check_in,
        check_out=check_out
    ))
    return checknums
