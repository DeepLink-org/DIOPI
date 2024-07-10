# Copyright (c) 2023, DeepLink.
import os
import argparse
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
new_content.append('''
/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions.h>
#include <diopi/functions_ext.h>
#include <diopi/functions_mmcv.h>
#include <dlfcn.h>
#include <dyn_helper.hpp>

#include <cstdio>

static void* handle;
const static char* diopiFile = "libdiopi_real_impl.so";
static void __attribute__((constructor)) diopi_init() { handle = dynLoadFile(diopiFile); }
static void __attribute__((destructor)) diopi_fini() { dlclose(handle); }

''')


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
            else:
                arg_type, arg = get_func_arg(temp_content)

            for args in arg_type:
                new_content.append(args)

            new_content.append("    " + 'func = reinterpret_cast<decltype(func)>(dlsym(handle, "' + func_name + '"));\n')
            new_content.append("    " + "if (func != nullptr) {\n")
            new_content.append("    " + "    return (*func)" + arg + ";\n")
            new_content.append("    " + "} else {\n")
            new_content.append("    " + "    printf(\"[wrap_func] %s not implemented!\\n\", \"" + func_name + "\");\n")
            if row.startswith("DIOPI_RT_API"):
                new_content.append("    " + "    return \"" + func_name + " not implemented!\";\n")
            else:
                new_content.append("    " + "    return diopiErrorOccurred;\n")
            new_content.append("    " + "}\n")
            new_content.append("}\n")
            new_content.append("\n")


def debugat():
    # rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    rank = 0
    if rank == 0:
        import os
        import ptvsd
        import socket

        pid1 = os.getpid()

        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(hostname, ip, flush=True)
        host = ip  # or "localhost"
        host = "127.0.0.1"
        port = 12346
        print("cwd is:", os.getcwd(), flush=True)
        ptvsd.enable_attach(address=(host, port), redirect_output=False)
        print("-------------------------print rank,:", rank, "pid1:", pid1, flush=True)
        ptvsd.wait_for_attach()


# debugat()



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DIOPI adaptor source files"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="output generated dynamic call file",
    )
    args = parser.parse_args()
    return args

op_header_files = ['functions.h', 'functions_mmcv.h', 'functions_ext.h']

if __name__ == '__main__':
    args = parse_args()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    protodir = file_dir + "/../../../proto/include/diopi/"
    for fname in op_header_files:
      with open(os.path.join(protodir, fname), 'r')as f:
          content = f.readlines()
          print(f"generate for {fname}")
          gen_wrapper_func(content)


    print(f"generate {args.output_file}")
    out_dir = os.path.dirname(args.output_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output_file, 'w') as f:
        for row in new_content:
            f.write(row)
    print("finish codegen")
