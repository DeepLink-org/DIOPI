# Copyright (c) 2023, DeepLink.
import os
# CONVERT
#
# DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
#                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2);
# TO
# DIOPI_API diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
#                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
#     diopiError_t (*func) (diopiContextHandle_t, diopiTensorHandle_t,
#         diopiConstTensorHandle_t, diopiConstTensorHandle_t);
#     func = dlsym(handle, "diopiBmm");
#     return (*func)(ctx, out, input, mat2);
# }

new_content = []
new_content.append('/**\n\
 * @file\n\
 * @author DeepLink\n\
 * @copyright  (c) 2023, DeepLink.\n\
 */\n\
#include <diopi/functions.h>\n\
#include <diopi/functions_mmcv.h>\n\
#include <stdio.h>\n\
#include <dlfcn.h>\n\
\n\
static void* handle;\n\
\n\
static void\n\
__attribute__ ((constructor))\n\
diopi_init(void) {\n\
    handle = dlopen("libdiopi_real_impl.so", RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);\n\
    printf("diopi dyload init\\n");\n\
    if (!handle) {\n\
        fprintf (stderr, "%s ", dlerror());\n\
    }\n\
}\n\
\n\
static void\n\
__attribute__ ((destructor))\n\
diopi_fini(void)\n\
{\n\
dlclose(handle);\n\
}\n\
\n')


def get_func_arg(content):
    arg = "("
    new_content = []
    arg_type = "    diopiError_t (*func)"
    for row in content:
        idx0 = 0
        idx2 = row.find(",")
        while (idx2 != -1):
            idx1 = row.rfind(" ", idx0, idx2)
            idx_find_star = row.find("*", idx1, idx2)  # handle the case like diopiSize_t *
            idx1 = idx1 + 1 if idx_find_star != -1 else idx1
            arg += row[idx1 + 1:idx2] + ', '
            idx1 = idx1 + 1 if idx_find_star != -1 else idx1
            arg_type += row[idx0:idx1] + ','
            idx0 = idx2 + 1
            idx2 = row.find(",", idx0)
        if row == '(':
            arg_type += row
        new_content.append(arg_type + '\n')
        arg_type = "        "

    idx2 = row.find(")")
    idx1 = row.rfind(" ", idx0, idx2)
    arg += row[idx1 + 1:idx2] + ')'
    new_content[-1] = new_content[-1].replace('\n', row[idx0:idx1] + ');\n')
    return new_content, arg

def gen_wrapper_func(content):
    for idx, row in enumerate(content):
        if row.startswith("DIOPI"):
            temp_content = []
            idx1 = row.find("(")
            idx0 = row.rfind(" ", 0, idx1)
            func_name = row[idx0 + 1: idx1]  # idx0 ~ idx1 is func name
            temp_content.append(row[idx1:-1])
            idx2 = row.find(")")
            if idx2 != -1:
                new_content.append(row.replace(";", " {"))
            else:
                new_content.append(row)
            while idx2 == -1:
                row1 = content[idx + 1]
                idx2 = row1.find(")")
                if idx2 != -1:
                    new_content.append(row1.replace(";", " {"))
                else:
                    new_content.append(row1)
                temp_content.append(row1.lstrip())
                idx += 1

            if row.startswith("DIOPI_RT_API"):
                arg_type = ["    const char* (*func)();\n"]
                arg = "()"
                for args in arg_type:
                    new_content.append(args)
                new_content.append("    " + 'func = reinterpret_cast<decltype(func)>(dlsym(handle, "' + func_name + '"));\n')
                new_content.append("    " + "if (func != NULL) {\n")
                new_content.append("    " + "    return (*func)" + arg + ";\n")
                new_content.append("    " + "} else {\n")
                new_content.append("    " + "    printf(\"[wrap_func] %s not implemented!\\n\", \"" + func_name + "\");\n")
                new_content.append("    " + "    return \"" + func_name + " not implemented!\";\n")
                new_content.append("    " + "}\n")
                new_content.append("}\n")
                new_content.append("\n")
            else:
                arg_type, arg = get_func_arg(temp_content)
                for args in arg_type:
                    new_content.append(args)
                new_content.append("    " + 'func = reinterpret_cast<decltype(func)>(dlsym(handle, "' + func_name + '"));\n')
                new_content.append("    " + "if (func != NULL) {\n")
                new_content.append("    " + "    return (*func)" + arg + ";\n")
                new_content.append("    " + "} else {\n")
                new_content.append("    " + "    printf(\"[wrap_func] %s not implemented!\\n\", \"" + func_name + "\");\n")
                new_content.append("    " + "    return diopiErrorOccurred;\n")
                new_content.append("    " + "}\n")
                new_content.append("}\n")
                new_content.append("\n")

if __name__ == '__main__':
    print("open functions.h")
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_cur_dir, '../proto/include/diopi/functions.h'), 'r')as f:
        content = f.readlines()
    print("generate for functions.h")
    gen_wrapper_func(content)
    print("open functions_mmcv.h")
    with open(os.path.join(_cur_dir, '../proto/include/diopi/functions_mmcv.h'), 'r') as f:
        content_mmcv = f.readlines()
    print("generate for functions_mmcv.h")
    gen_wrapper_func(content_mmcv)
    os.system("rm -f wrap_func.cpp")
    print("generate wrap_func.cpp")
    with open('wrap_func.cpp', 'w') as f:
        for row in new_content:
            f.write(row)
    print("finish codegen")
