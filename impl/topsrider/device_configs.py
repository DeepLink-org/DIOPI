from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'batch_norm': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 64, 32, 32)),],
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=['conv2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 7, 12, 13)),Skip((6, 16, 19, 8)),Skip((6, 27, 12, 8)),Skip((2, 256, 200, 304)),Skip((2, 2048, 64, 64)),Skip((2, 2048, 1, 1)),Skip((2, 256, 200, 304)),],
                },
            ]
        ),
    ),

    'relu': dict(
        name=['relu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.int16),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=['max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'pointwise_op': dict(
        name=['abs', 'exp', 'floor', 'sqrt', 'rsqrt', 'ceil', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_int_without_inplace': dict(
        name=['abs', 'exp', 'sqrt', 'rsqrt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_abs_input': dict(
        name=['log', 'log2', 'log10', 'sqrt', 'rsqrt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pow': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'pow_broadcast': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 1, 128)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((8, 16, 1)),],
                },
            ]
        ),
    ),

    'pow_broadcast_inplace': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((5, 2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_broadcast': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((8, 16, 1)),],
                },
            ]
        ),
    ),

    'pointwise_binary_broadcast_inplace': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((5, 2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'bitwise_op': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'div': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'div_broadcast': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((8, 16, 1)),],
                },
            ]
        ),
    ),

    'sub_scalar': dict(
        name=['sub'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'clamp_max_scalar': dict(
        name=['clamp_max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'clamp_min_scalar': dict(
        name=['clamp_min'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((182,)),Skip((384, 128)),Skip((1, 242991, 2)),Skip((2, 4, 100, 152)),],
                },
            ]
        ),
    ),

    'clamp_max_tensor': dict(
        name=['clamp_max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((182,)),Skip((384, 128)),Skip((3, 242991, 2)),Skip((2, 4, 100, 152)),],
                },
            ]
        ),
    ),

    'clamp_min_tensor': dict(
        name=['clamp_min'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((182,)),Skip((384, 128)),Skip((3, 242991, 2)),Skip((2, 4, 100, 152)),],
                },
            ]
        ),
    ),

    'fill': dict(
        name=['fill_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'max_min_all': dict(
        name=['min', 'max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'select': dict(
        name=['select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'index_select': dict(
        name=['index_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'index_select_not_float': dict(
        name=['index_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int32),Skip(Dtype.int16),],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=['log_softmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'soft_max': dict(
        name=['softmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'tril': dict(
        name=['tril'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'join': dict(
        name=['cat'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'cat_diff_size': dict(
        name=['cat'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'sort': dict(
        name=['sort'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((11400,)),Skip((12, 8)),Skip((8, 12, 9)),Skip((4, 4, 16, 20)),Skip((4, 4, 16, 2, 20)),Skip((24180,)),],
                },
            ]
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 1536, 950)),Skip((16, 8)),Skip((660, 6, 49, 32)),],
                },
            ]
        ),
    ),

    'leaky_relu': dict(
        name=['leaky_relu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'reciprocal': dict(
        name=['reciprocal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'bitwise_not_int': dict(
        name=['bitwise_not'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'expand': dict(
        name=['expand'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.int16),],
                },
            ]
        ),
    ),

    'gather': dict(
        name=['gather'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'arange': dict(
        name=['arange'],
        para=dict(
            start=[Skip(0),Skip(0),Skip(-4),Skip(0.1),Skip(10),Skip(2.3),Skip(True),Skip(-20),Skip(90),Skip(0.001),],
            end=[Skip(91),Skip(128),Skip(5),Skip(0.5),Skip(10),Skip(2.3),Skip(100),Skip(False),Skip(-90),Skip(0.0001),],
            step=[Skip(13),Skip(1),Skip(1),Skip(0.1),Skip(True),Skip(0.5),Skip(2.1),Skip(0.5),Skip(-5.6),Skip(-1e-05),],
        ),
    ),

    'layer_norm': dict(
        name=['layer_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 5, 3, 5)),Skip((2, 3136, 128)),Skip((2, 64)),Skip((32,)),Skip((2, 5, 3, 5)),Skip((2, 16, 128)),],
                },
            ]
        ),
    ),

    'interpolate': dict(
        name=['interpolate'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 256, 25, 38)),Skip((2, 2, 16, 16)),Skip((2, 2, 16, 16)),Skip((2, 256, 13, 19)),Skip((3, 12, 14, 19)),Skip((2, 16, 1, 1)),Skip((2, 16, 15, 32)),],
                },
            ]
        ),
    ),

    'flip': dict(
        name=['flip'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.int16),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'cast_dtype': dict(
        name=['cast_dtype'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),],
                },
            ]
        ),
    ),

}
