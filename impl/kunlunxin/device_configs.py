from skip import Skip
import numpy as np

device_configs = {
    'batch_norm_nan': dict(
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

    'batch_norm_no_contiguous': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((32, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_backward_elemt': dict(
        name=['batch_norm_backward_elemt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['grad_out'],
                    "shape": [Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_elemt': dict(
        name=['batch_norm_elemt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
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
                    "shape": [Skip((2, 256, 200, 304)),],
                },
            ]
        ),
    ),

    'conv_2d_no_contiguous': dict(
        name=['conv2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 256, 200, 304)),Skip((2, 2048, 64, 64)),Skip((2, 2048, 1, 1)),Skip((2, 256, 200, 304)),],
                },
            ]
        ),
    ),

    'copy_all_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((192, 147)), Skip((192, 147, 2)), Skip((2, 12, 38, 45, 3))],
                },
            ]
        )
    ),

    'cross_entropy': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024, 81)),Skip((3, 9)),Skip((64, 9)),Skip((5, 9, 12, 4)),Skip((0, 16)),Skip((0, 5, 6)),Skip((4, 6, 0, 3)),],
                },
            ]
        ),
    ),

    'index_put_acc_three_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_put_acc_two_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_put_acc_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
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
                    "shape": [Skip((0,)),Skip((0,15)),Skip((5,0,13)),],
                },
            ]
        ),
    ),

    'nll_loss': dict(
        name=['nll_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((200,79)),Skip((2,150, 128, 128)),Skip((0, 6, 1, 3)),Skip((0, 5, 6, 1, 3)),Skip((5, 16, 0)),Skip((3, 80, 25, 24, 5)),Skip((1,)),Skip((2,)),Skip((79,)),Skip((180,80)),],
                },
            ]
        ),
    ),

    'nll_loss_empty_tensor': dict(
        name=['nll_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)),Skip((16, 0)),Skip((5, 0, 5, 6, 0, 3)),Skip((4, 0, 8, 3)),],
                },
            ]
        ),
    ),

    'pointwise_binary': dict(
        name=['add','sub'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8), Skip(np.int8),Skip(np.int16),],
                },
            ]
        ),
    ),

    'pointwise_binary_broadcast': dict(
        name=['add', ],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8), Skip(np.int8),Skip(np.int16),],
                },
            ]
        ),
    ),

    'pointwise_binary_diff_dtype_without_bool': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64),Skip(np.float32),Skip(np.float16),Skip(np.int32),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),Skip(np.float32),],
                },
            ]
        ),
    ),

    'pointwise_binary_dtype_bool': dict(
        name=['add', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['add', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8), Skip(np.int8),Skip(np.int16),],
                },
            ]
        ),
    ),

    'reduce_op': dict(
        name=['mean'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((16,0,9)),Skip((0,)),Skip((0,2)),],
                },
            ]
        ),
    ),

    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'reduce_partial_op_4': dict(
        name=['sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                    "dtype": [Skip(np.int8),Skip(np.uint8),Skip(np.int16),Skip(np.int32),Skip(np.int64),],
                },
            ]
        ),
    ),

    'sort': dict(
        name=['sort'],
        para=dict(dim=[Skip(1),]),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int8),Skip(np.uint8),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'sort_same_value': dict(
        name=['sort'],
        para=dict(
            dim=[Skip(1),],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int8),Skip(np.uint8),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.float16),Skip(np.float32),Skip(np.float64),],
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
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_with_alpha': dict(
        name=['add', 'sub'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 3)),Skip((2, 2, 4, 3)),],
                },
            ]
        ),
    ),

    'pointwise_binary_with_alpha_bool': dict(
        name=['add'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 3)),Skip((2, 2, 4, 3)),],
                },
            ]
        ),
    ),

    'norm': dict(
        name=['norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'randperm': dict(
        name=['randperm'],
        para=dict(
            n=[Skip(2),Skip(1999),Skip(640000),Skip(0),Skip(1),],
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),],
                },
            ]
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'copy': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip(()), Skip((8,)), Skip((12,)), Skip((192, 147)), Skip((1, 1, 384)), Skip((1, 192, 147, 2)),
                              Skip((0,)), Skip((12, 0, 9)), Skip((0, 2))],
                },
            ]
        )
    ),

    'copy_input_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((12, 2)), Skip((12, 1, 12)), Skip((2, 38, 45, 2))),
                }
            ]
        )
    ),

    'copy_other_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((6, 5, 384)), Skip((2, 4, 38, 45)))
                }
            ]
        )
    ),

    'normal': dict(
        name=['normal'],
        para=dict(
            mean=[Skip(-1),Skip(-0.5),Skip(0),Skip(0.1),Skip(2),Skip(True),Skip(False),Skip(0.2),Skip(-2),Skip(0),],
            std=[Skip(0),Skip(0.5),Skip(1),Skip(2.3),Skip(3),Skip(True),Skip(True),Skip(0.5),Skip(0),Skip(3),],
            size=[Skip(()),Skip((1280,)),Skip((32, 160)),Skip((320, 8)),Skip((32, 80)),Skip((2, 2, 20, 16)),Skip((320, 2, 3, 3)),Skip((0,)),Skip((4, 0)),Skip((2, 0, 9)),],
        ),
    ),

    'normal_': dict(
        name=['normal_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'normal_std_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'normal_mean_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'normal_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float64),],
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
                    "shape": [Skip(()), Skip((0,)), Skip((4, 0)), Skip((5, 0, 7))],
                    "dtype": [Skip(np.float64),Skip(np.float16),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),Skip(np.uint8),Skip(np.int8),Skip(np.int8),],
                },
                {
                     "ins": ['out'],
                     "shape": [Skip(()), Skip((0,)), Skip((4, 0)), Skip((5, 0, 7))],
                     "dtype": [Skip(np.uint8), Skip(np.bool_),
                              Skip(np.float64), Skip(np.int8),
                              Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
                },

            ]
        ),
    ),

}