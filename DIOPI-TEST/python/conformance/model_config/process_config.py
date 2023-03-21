import cv_config
import seg_config
import os
import det_config
import other_config
import torch

unary_op = {'input': 'tensor'}
unary_inp_op = {'input': 'tensor', 'inplace': 'para/key'}
binary_op = {'input': 'tensor', 'other': 'tensor'}
reduce_op = {'input': 'tensor', 'dim': 'para/key'}

func_para = dict(
    relu=unary_inp_op,
    floor=unary_inp_op,
    neg=unary_inp_op,
    reciprocal=unary_inp_op,
    abs=unary_inp_op,
    leaky_relu=unary_inp_op,
    nonzero=unary_op,
    sqrt=unary_op,
    sin=unary_op,
    cos=unary_op,
    log=unary_op,
    log2=unary_op,
    bitwise_not=unary_op,
    exp=unary_op,
    erfinv=unary_inp_op,
    logical_and=binary_op,
    matmul=binary_op,
    maximum=binary_op,
    minimum=binary_op,
    ge=binary_op,
    le=binary_op,
    lt=binary_op,
    gt=binary_op,
    all=reduce_op,
    any=reduce_op,
    max=reduce_op,
    min=reduce_op,
    argmax=reduce_op,
    std={'input': 'tensor', 'dim': 'para', 'unbiased': 'para'},
    remainder={'input': 'tensor/scalar', 'other': 'tensor/scalar'},
    conv_transpose2d={"input": "tensor", "weight": "tensor", "bias": "tensor/none", "stride": "para/key",
                      "padding": "para/key", "output_padding": "para/key", "groups": "para/key", "dilation": "para/key"},
    conv2d={"input": "tensor/grad", "weight": "tensor/grad", "bias": "tensor/none/grad",
            "stride": "para", "padding": "para", "dilation": "para", "groups": "para"},
    batch_norm={"input": "tensor/grad", "running_mean": "tensor", "running_var": "tensor", "weight": "tensor/none/grad",
                "bias": "tensor/none/grad", "training": "para", "momentum": "para", "eps": "para"},
    max_pool2d={"input": "tensor/grad", "kernel_size": "para", "stride": "para/key", "padding": "para/key",
                "dilation": "para/key", "ceil_mode": "para/key", "return_indices": "par/key"},
    adaptive_avg_pool2d={"input": "tensor/grad", "output_size": "para"},
    linear={"input": "tensor/grad", "weight": "tensor/grad", "bias": "tensor/none/grad"},
    cross_entropy={"input": "tensor/grad", "target": "tensor", "weight": "tensor/none/grad", "size_average": "para/key",
                   "ignore_index": "para/key", "reduce": "para/key", "reduction": "para/key", "label_smoothing": "para/key"},
    add={"input": "tensor", "other": "tensor/scalar", "alpha": "para/key"},
    sum={"input": "tensor", "dim": "para/key", "dtype": "para/key"},
    mean={"input": "tensor", "dim": "para/key", "dtype": "para/key"},
    mul={"input": "tensor", "other": "tensor/scalar"},
    div={"input": "tensor", "other": "tensor/scalar", "rounding_mode": "para/key"},
    randperm={"n": "para", "dtype": "para/key"},
    sgd={'param", "param_grad': "tensor", "buf": "tensor", "nesterov": "para/key", "lr": "para/key", "momentum": "para/key",
         "weight_decay": "para/key", "dampening": "para/key"},
    cat={'tensors': "tensorlist", 'dim': "para/key"},
    avg_pool2d={"input": "tensor/grad", "kernel_size": "para", "stride": "para/key", "padding": "para/key", "ceil_mode": "para/key",
                "count_include_pad": "para/key", "divisor_override": "para/key"},
    sigmoid={'input': "tensor/grad"},
    hardtanh={'input': "tensor/grad", 'min_val': "para/key", "max_val": "para/key", "inplace": "para/key"},
    linspace={'start': "para", 'end': "para", "steps": 'para', 'dtype': 'para/key'},
    pad={'input': "tensor", 'pad': 'para', 'mode': "para/key", 'value': 'para'},
    transpose={'input': 'tensor', 'dim0': 'para', 'dim1': 'para'},
    dropout={'input': 'tensor', 'p': 'para/key', 'training': 'para/key', 'inplace': 'para/key'},
    dropout2d={'input': 'tensor', 'p': 'para/key', 'training': 'para/key', 'inplace': 'para/key'},
    arange={'start': "para/key", "end": "para", "step": "para/key", 'dtype': 'para/key'},
    one_hot={'input': 'tensor', 'num_classes': 'para/key'},
    layer_norm={'input': 'tensor/grad', 'normalized_shape': 'para/key', 'weight': 'tensor/none/grad', 'bias': 'tensor/none/grad', 'eps': 'para/key'},
    permute={'input': 'tensor', 'dims': 'para/key'},
    flip={'input': 'tensor', 'dims': 'para'},
    group_norm={'input': 'tensor/grad', 'num_groups': 'para', 'weight': 'tensor/none/grad', 'bias': 'tensor/none/grad', 'eps': 'para/key'},
    softmax={'input': 'tensor', 'dim': 'para/key', 'dtype': 'para/key'},
    gelu={'input': 'tensor/grad', 'approximate': 'para/key'},
    roll={'input': 'tensor', 'shifts': 'para/key', 'dims': 'para/key'},
    sub={"input": "tensor", "other": "tensor/scalar", "alpha": "para/key"},
    ne={"input": "tensor", "other": "tensor/scalar", "inplace": 'para/key'},
    eq={"input": "tensor", "other": "tensor/scalar", "inplace": 'para/key'},
    masked_fill={'input': 'tensor', 'mask': 'tensor', 'value': 'tensor/scalar', 'inplace': 'para/key'},
    log_softmax={'input': 'tensor', 'dim': 'para/key', 'dtype': 'para/key'},
    unfold={'input': 'tensor/grad', 'dimension': 'para', 'size': 'para', 'step': 'para'},
    im2col={'input': 'tensor', 'kernel_size': 'para', 'dilation': 'para/key', 'padding': 'para/key', 'stride': 'para/key'},
    norm={'input': 'tensor', 'p': 'para', 'dim': 'para/key', 'keepdim': 'para/key', 'dtype': 'para/key'},
    stack={'tensors': "tensorlist", 'dim': "para/key"},
    clamp={'input': 'tensor', 'min': 'tensor/scalar/key', 'max': 'tensor/scalar/key', 'inplace': 'para/key'},
    addcmul={'input': 'tensor', 'tensor1': 'tensor', 'tensor2': 'tensor', 'value': 'para/key'},
    addcdiv={'input': 'tensor', 'tensor1': 'tensor', 'tensor2': 'tensor', 'value': 'para/key'},
    expand={'input': 'tensor', 'size': 'para'},
    tanh={'input': "tensor/grad"},
    pow={'input': 'tensor/para', 'exponent': 'tensor/para'},
    index_select={'input': 'tensor', 'dim': 'para', 'index': 'tensor'},  # to check manually index not out of range
    split={'tensor': 'tensor', 'split_size_or_sections': 'para', 'dim': 'para/key'},
    mse_loss={'input': 'tensor', 'target': 'tensor', 'size_average': 'para/key', 'reduce': 'para/key', 'reduction': 'para/key'},
    binary_cross_entropy_with_logits={'input': 'tensor', 'target': 'tensor', 'weight': 'tensor/none', 'size_average': 'para/key',
                                      'reduce': 'para/key', 'reduction': 'para/key', 'pos_weight': 'tensor/none'},
    interpolate={'input': 'tensor', 'size': 'para/key', 'scale_factor': 'para/key', 'mode': 'para/key', 'align_corners': 'para/key'},
    where={'condition': 'tensor', 'input': 'tensor/scalar', 'other': 'tensor/scalar'},
    sort={'input': 'tensor', 'dim': 'para/key', 'descending': 'para/key', 'stable': 'para/key'},
    uniform={'input': 'tensor', 'start': 'para/key', 'end': 'para/key'},
    fill_={'input': 'tensor', 'value': 'scalar'},
    unique={'input': 'tensor', 'sorted': 'para/key', 'return_inverse': 'para/key', 'return_counts': 'para/key', 'dim': 'para/key'},
    topk={'input': 'tensor', 'k': 'para', 'dim': 'para/key', 'largest': 'para/key', 'sorted': 'para/key'},
    adamw={'param", "param_grad': "tensor", 'exp_avg", "exp_avg_sq", "max_exp_avg_sq': "tensor", 'step': 'para',
           "amsgrad": "para/key", "beta1": "para/key", "beta2": "para/key", "lr": "para/key", "weight_decay": "para/key", "eps": "para/key"},
    cdist={'x1': 'tensor/grad', 'x2': 'tensor', 'p': 'para/key', 'compute_mode': 'para/key'},
    bmm={'input': 'tensor', 'mat2': 'tensor'},
    cumsum={'input': 'tensor', 'dim': 'para', 'dtype': 'para/key'},
    adam={'param", "param_grad': "tensor", 'exp_avg", "exp_avg_sq", "max_exp_avg_sq': "tensor", 'step': 'para',
          "amsgrad": "para/key", "beta1": "para/key", "beta2": "para/key", "lr": "para/key", "weight_decay": "para/key", "eps": "para/key"},
    embedding={'input': 'tensor', 'weight': 'tensor/grad', "padding_idx": 'para/key', 'max_norm': 'para/key', "norm_type": 'para/key',
               'scale_grad_by_freq': 'para/key', 'sparse': 'para/key'},
    smooth_l1_loss={'input': 'tensor/grad', 'target': 'tensor', 'size_average': 'para/key', 'reduce': 'para/key', 'reduction': 'para/key', 'beta': 'para/key'},
    adadelta={'param", "param_grad': "tensor", 'square_avg", "acc_delta': 'tensor', 'lr': 'para', 'rho': 'para', 'eps': 'para', 'weight_decay': 'para'},
    triangular_solve={'input': 'tensor/grad', 'A': 'tensor/grad', 'upper': 'para/key', 'transpose': 'para/key', 'unitriangular': 'para/key'},
    gather={'input': 'tensor', 'dim': 'para', 'index': 'tensor'},  # to check manually index not out of range
    conv3d={"input": "tensor/grad", "weight": "tensor/grad", "bias": "tensor/none/grad",
            "stride": "para/key", "padding": "para/key", "dilation": "para/key", "groups": "para/key"},
    max_pool3d={"input": "tensor/grad", "kernel_size": "para", "stride": "para/key", "padding": "para/key",
                "dilation": "para/key", "ceil_mode": "para/key", "return_indices": "par/key"},
    adaptive_avg_pool3d={"input": "tensor/grad", "output_size": "para"},
    cholesky_ex={"input": "tensor/grad", "upper": "para/key", "check_errors": 'para/key'},
    normal={'mean': 'tensor/para', 'std': 'tensor/para', 'size': 'para/key'},
    normal_={'size': 'para/key', 'mean': 'tensor/para', 'std': 'tensor/para'},
)

convert_name = {'iadd': "add", 'radd': "add", 'add_': "add", 'rmul': 'mul', 'truediv': 'div', 'rtruediv': 'div',
                'mul_': 'mul', 'addcmul_': 'addcmul', 'addcdiv_': 'addcdiv', 'uniform_': 'uniform', 'rand': 'uniform',
                'and': 'logical_and', 'sub_': 'sub', 'div_': 'div', 'imul': 'mul', 'clamp_': 'clamp', 'sigmoid_': 'sigmoid',
                'itruediv': 'div', 'invert': 'bitwise_not', 'rsub': 'sub', 'expand_as': 'expand', 't': 'transpose',
                'erfinv_': 'erfinv', 'floordiv': 'div', 'rpow': 'pow', 'isub': 'sub', 'sqrt_': 'sqrt', 'masked_fill_': 'masked_fill',
                'mod': 'remainder', 'cholesky': 'cholesky_ex'}
inplace_tag = ['iadd', 'imul', 'mul_', 'sub_', 'div_', 'clamp_', 'sigmoid_', 'itruediv', 'erfinv_', 'isub', 'masked_fill_']
interface_tag = {"sgd": "CustomizedTest", "adamw": "CustomizedTest", 'im2col': 'CustomizedTest', 'adadelta': 'CustomizedTest',
                 "split": "torch", 'cholesky_ex': 'torch.linalg', "adam": "CustomizedTest"}
no_output_ref = ['randperm', 'uniform', 'dropout', 'dropout2d', 'normal', 'normal_']
saved_args = {"sigmoid": "0", 'softmax': '0', 'log_softmax': '0', 'tanh': '0', 'cholesky_ex': '0', 'cdist': '0',
              'triangular_solve': '0'}
requires_backward = {'cholesky_ex': '0'}
gen_func = {'cholesky_ex/input': 'Genfunc.sym_mat', 'normal/std': 'Genfunc.positive',
            'adadelta/square_avg", "acc_delta': 'Genfunc.positive'}

tensor_vide = "                    "
para_vide = "            "
key_vide = "        "
seq_name = ['cat', 'stack']
ignore_list = ['getitem', 'relu_', 'setitem', 'get_rank', 'get_world_size', 'barrier',
               'load', 'broadcast', 'repeat', 'all_reduce', 'meshgrid', 'eye', 'conj',
               'diagonal', 'grid_sample', 'einsum']


def toDtype(dtype, tensor_para, gen_func=None):
    gen_fn_str = 'Genfunc.randn'
    if dtype == 'torch.cuda.FloatTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.float32],\n')
    elif dtype == 'torch.cuda.DoubleTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.float32],\n')
    elif dtype == 'torch.cuda.LongTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.int64],\n')
        gen_fn_str = 'Genfunc.randint'
    elif dtype == 'torch.cuda.BoolTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.bool],\n')
        gen_fn_str = 'Genfunc.mask'
    elif dtype == 'torch.cuda.IntTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.int32],\n')
        gen_fn_str = 'Genfunc.randint'
    elif dtype == 'torch.cuda.ByteTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.uint8],\n')
        gen_fn_str = 'Genfunc.randint'
    elif dtype == 'torch.cuda.CharTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.int8],\n')
    elif dtype == 'torch.cuda.HalfTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.float16],\n')
    elif dtype == 'torch.cuda.ShortTensor':
        tensor_para.append(tensor_vide + '"dtype": [Dtype.int16],\n')
    elif isinstance(dtype, list):
        dtype_list = [ele.replace("torch", "Dtype") for ele in dtype]
        dtype_list = list(set(dtype_list))
        tensor_para.append(tensor_vide + '"dtype": ' + str(dtype_list).replace("'", "") + ',\n')

    gen_fn = gen_fn_str if gen_func is None else gen_func
    tensor_para.append(tensor_vide + '"gen_fn": ' + gen_fn + ',\n')


def gen_config_code(config, file_name):
    content = config
    names = {}

    os.system(f"rm -f {file_name}.py")
    with open(f'{file_name}.py', 'a') as f:
        f.write("from ...config import Genfunc\n")
        f.write("from ...dtype import Dtype\n\n")
        f.write(file_name[file_name.find("/") + 1:] + " = {\n")

    for ele in content:
        name = ele[0]
        para_list = ele[3]
        kpara_list = ele[4]
        type_list = ele[2]
        if name == 'unfold' and ele[1] == 'torch.nn.functional':
            name = 'im2col'
        elif name == 'max' and len(type_list) == 2:
            name = 'maximum'
        elif name == 'min' and len(type_list) == 2:
            name = 'minimum'
        if name in convert_name.keys():
            name = convert_name[name]
        if name in ['sgd', 'adamw'] and not type_list:
            continue
        if name not in func_para.keys():
            if name not in ignore_list:
                print(f"%%%%%%% miss definition for {name} while generate {file_name}.py %%%%%%%%%%\n")
            continue
        if para_list and len(para_list[0]) > 500:
            print(f"Warning: too many {len(para_list[0])} for {name} while generate {file_name}.py %%%%%%%%%%\n")

        para = []
        tensor_para = []
        idx = 0
        type_idx = 0

        if name not in names.keys():
            config = ["    '" + name + "': dict(\n"]
            names.update({name: 0})
        else:
            names[name] += 1
            config = ["    '" + name + "_" + str(names[name]) + "': dict(\n"]
        config.append(key_vide + 'name=["' + name + '"],\n')

        if ele[0] in inplace_tag:
            config.append(key_vide + 'is_inplace=[True],\n')
        if name in no_output_ref:
            config.append(key_vide + 'no_output_ref=True,\n')
        elif name in interface_tag.keys():
            config.append(key_vide + 'interface=["' + interface_tag[name] + '"],\n')
        elif ele[1] != 'torch.nn.functional':
            config.append(key_vide + 'interface=["' + ele[1] + '"],\n')

        if name in saved_args.keys():
            config.append(key_vide + 'saved_args=dict(output=' + saved_args[name] + '),\n')
        if name in requires_backward.keys():
            config.append(key_vide + 'requires_backward=[' + requires_backward[name] + '],\n')

        for k, v in func_para[name].items():
            if idx >= len(para_list) + len(kpara_list):
                break
            if k == "index":
                print(f"Warning: need to to check manually index not out of range for {name} in {file_name}.py %%%%%%%%%%\n")

            is_para = True
            if "tensor" in v:
                if name in ['sgd', 'adamw', 'adam', 'adadelta']:
                    type_idx = 0
                    idx = 0
                    is_para = False
                elif idx >= len(para_list):
                    if "scalar/key" not in v:
                        assert "none" in v, "tensor can not be None"
                        continue
                elif isinstance(para_list[idx][0], tuple):
                    is_para = False

                if not is_para:
                    tensor_para.append(para_vide + "    {\n" + tensor_vide + '"ins": ["' + str(k) + '"],\n')
                    if "grad" in v and type_idx < len(type_list):
                        tensor_para.append(tensor_vide + '"requires_grad": [True],\n')

                    tensor_para.append(tensor_vide + '"shape": ' + str(para_list[idx]) + ",\n")
                    if ele[0] == 'rand':
                        toDtype(kpara_list['dtype'], tensor_para)
                    elif type_idx < len(type_list):
                        gen_func_key = name + '/' + k
                        gen_fn = gen_func[gen_func_key] if gen_func_key in gen_func.keys() else None
                        toDtype(type_list[type_idx], tensor_para, gen_fn)
                        type_idx += 1
                    tensor_para.append(para_vide + "    },\n")

            if is_para:
                if k in kpara_list.keys():
                    if name in ['sgd', 'adamw', 'adam', 'adadelta'] and (not isinstance(kpara_list[k], list) or len(kpara_list[k]) == 1):
                        kpara_list[k] = kpara_list[k][0] if isinstance(kpara_list[k], list) else kpara_list[k]
                        para.append(para_vide + str(k) + "=[" + str(kpara_list[k]) + f" for i in range({len(para_list[0])})],\n")
                    else:
                        if k == 'size' and type(kpara_list[k][0]) == torch.Size:  # convert torch.Size to tuple
                            kpara_list[k] = [tuple(e) for e in kpara_list[k]]
                        if not isinstance(kpara_list[k], list):
                            kpara_list[k] = [kpara_list[k]]
                        if k == "dtype":
                            kpara_list[k] = str(kpara_list[k]).replace("torch.", "Dtype.").replace("'", "")
                        para.append(para_vide + str(k) + "=" + str(kpara_list[k]).replace('-inf', 'float("-inf")') + ",\n")
                elif idx < len(para_list):
                    if name in ['permute', 'expand'] and not isinstance(para_list[idx][0], (tuple, list)):
                        dims_list = []
                        for i in range(len(para_list[idx])):
                            dims = [para_list[j][i] for j in range(idx, len(para_list))]
                            dims_list.append(tuple(dims))
                        para_list[idx] = dims_list

                    if name == 'arange' and idx == len(para_list) - 1 and k == 'start':
                        k = 'end'

                    if k == 'size' and type(para_list[idx][0]) == torch.Size:  # convert torch.Size to tuple
                        para_list[idx] = [tuple(e) for e in para_list[idx]]

                    para.append(para_vide + str(k) + "=" + str(para_list[idx]).replace('-inf', 'float("-inf")') + ",\n")
                elif name in ['adamw', 'adam'] and k == 'step':
                    step_list = [i + 1 for i in range(len(para_list[0]))]
                    idx -= 1
                    para.append(para_vide + str(k) + "=" + str(step_list) + ",\n")
                elif "key" not in v and name != 'arange':
                    print(f"%%%%%%% miss '{k}' in {name} op while generate {file_name}.py %%%%%%%%%%\n")
                    continue
                else:
                    continue
            idx += 1

        if para:
            config.append(key_vide + "para=dict(\n")
            for e in para:
                config.append(e)
            config.append(key_vide + "),\n")
        if ele[0] == 't':
            config.append(key_vide + "para=dict(\n")
            length = len(para_list[0])
            config.append(para_vide + 'dim0=[0 for i in range(' + str(length) + ")],\n")
            config.append(para_vide + 'dim1=[1 for i in range(' + str(length) + ")],\n")
            config.append(key_vide + "),\n")
        if tensor_para:
            config.append(key_vide + "tensor_para=dict(\n")
            config.append(para_vide + "args=[\n")
            for e in tensor_para:
                config.append(e)
            config.append(para_vide + "],\n")
            if name in seq_name:
                config.append(para_vide + "seq_name='tensors',\n")
            config.append(key_vide + "),\n")
        config.append("    ),\n")
        config.append("\n")

        with open(f'{file_name}.py', 'a') as f:
            for row in config:
                f.write(row)
    with open(f'{file_name}.py', 'a') as f:
        f.write("}\n")


if __name__ == '__main__':
    cv_config_dict = {"resnet50_config": cv_config.resnet50_8xb32_in1k_config,
                      'resnet101_config': cv_config.resnet101_8xb32_in1k_config,
                      'densenet_config': cv_config.densenet121_4xb256_in1k_config,
                      'seresnet50_config': cv_config.seresnet50_8xb32_in1k_config,
                      'efficientnet_config': cv_config.efficientnet_b2_8xb32_in1k_config,
                      "mobilenet_v2_config": cv_config.mobilenet_v2_8xb32_in1k_config,
                      "repvgg_config": cv_config.repvgg_A0_4xb64_coslr_120e_in1k_config,
                      "shufflenet_v2_config": cv_config.shufflenet_v2_1x_16xb64_in1k_config,
                      "swin_transformer_config": cv_config.swin_base_16xb64_in1k_config,
                      "vit_config": cv_config.vit_base_p16_pt_64xb64_in1k_224_config,
                      "vgg16_config": cv_config.vgg16_8xb32_in1k_config,
                      "inceptionv3_config": cv_config.inception_v3_8xb32_in1k_config}
    det_config_dict = {"faster_rcnn_r50_config": det_config.faster_rcnn_r101_fpn_1x_coco_config,
                       "retinanet_config": det_config.retinanet_r50_fpn_1x_coco_config,
                       "ssd300_config": det_config.ssd300_coco_config,
                       "yolov3_config": det_config.yolov3_d53_320_273e_coco_config,
                       "cascade_rcnn_config": det_config.cascade_rcnn_r50_fpn_1x_coco_config,
                       'atss_config': det_config.atss_r50_fpn_1x_coco_config,
                       "fcos_config": det_config.fcos_r50_caffe_fpn_gn_head_1x_coco_config,
                       "mask_rcnn_config": det_config.mask_rcnn_r50_fpn_1x_coco_config,
                       "solo_config": det_config.solo_r50_fpn_1x_coco_config,
                       "detr_config": det_config.detr_r50_8x2_150e_coco_config,
                       "centernet_config": det_config.centernet_resnet18_140e_coco_config}
    seg_config_dict = {"unet_config": seg_config.fcn_unet_s5_d16_4x4_512x1024_160k_cityscapes_config,
                       "fcn_config": seg_config.fcn_d6_r50_d16_512x1024_40k_cityscapes_config,
                       "deeplabv3_config": seg_config.deeplabv3_r50_d8_512x1024_40k_cityscapes_config,
                       "deeplabv3plus_config": seg_config.deeplabv3plus_r50_d8_512x1024_40k_cityscapes_config,
                       "pspnet_config": seg_config.pspnet_r50_d8_512x1024_40k_cityscapes_config,
                       "upernet_config": seg_config.upernet_r50_512x1024_40k_cityscapes_config}
    other_config_dict = {"stgcn_config": other_config.stgcn_80e_ntu60_xsub_keypoint_config,
                         "hrnet_config": other_config.hrnet_w32_coco_wholebody_512x512_config,
                         "deeppose_config": other_config.res50_coco_256x192_rle_config,
                         'crnn_config': other_config.crnn_mini_vgg_5e_mj_config,
                         'sar_config': other_config.sar_resnet31_parallel_decoder_5e_st_sub_mj_sub_sa_real_config,
                         'dbnet_config': other_config.dbnet_resnet18_fpnc_1200e_icdar2015_config,
                         'slowfast_config': other_config.slowfast_r50_16x8x1_22e_sthv1_rgb_config,
                         'tsn_config': other_config.tsn_r50_1x1x8_50e_sthv1_rgb_config}
    config_dict = other_config_dict
    for k, v in config_dict.items():
        gen_config_code(v, "other_configs/" + k)
