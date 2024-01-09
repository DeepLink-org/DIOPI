# Copyright (c) 2023, DeepLink.
import numpy as np
from skip import Skip

# topk llm used
# normal llm used
# norm llm used
# nll_loss llm used
# gather llm used
# fill_ llm used
# triu llm used

device_configs = {
    # temp for 910B
    'uniform': dict(
        name=['uniform'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),],
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=['batch_norm'],
        atol=1e-2,
        rtol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # Skip due to low precision
                    "dtype": [Skip(np.float16),],
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
                    # "dtype": [Skip(np.float16),],
                    # temp for 910B
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=['conv2d'],
        atol=1e-1,
        rtol=1e-2,
    ),

    'conv_2d_no_contiguous': dict(
        name=["conv2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.float32), Skip(np.float16), Skip(np.float64)],
                },
            ]
        ),
    ),

    'relu_no_contiguous': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32), Skip(np.float64)],
                },
            ],
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        atol=1e-3,
        rtol=1e-3,
    ),

    'gelu_specific': dict(
        name=['gelu'],
        atol=1e-3,
        rtol=1e-3,
    ),

    'max_pool2d_return_indices': dict(
        name=['max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=['adaptive_avg_pool2d'],
        atol=2e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3,16,8)), Skip((4,16,12)), Skip((2,144,65,65))],
                },
            ]
        ),
    ),

    'pointwise_op': dict(
        name=['erf'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((16, 0)),],
                },
            ]
        ),
    ),

    'pointwise_op_int_without_inplace': dict(
        name=['erf'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_uint8': dict(
        name=['erf'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),],
                },
            ]
        ),
    ),

    'pointwise_op_bool': dict(
        name=['erf'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'tanh_not_float': dict(
        name=['tanh'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'pow_tensor': dict( # llm used
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((20267, 80)),Skip((2, 128, 3072)),Skip((2, 512, 38, 38)),Skip((0,)),Skip((0, 4)),Skip((9, 0, 3)),],
                },
            ]
        ),
    ),

    'pow_tensor_only_0_1': dict( # llm used
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((20267, 80)),Skip((2, 128, 3072)),Skip((2, 512, 38, 38)),Skip((0,)),Skip((0, 4)),Skip((9, 0, 3)),],
                },
            ]
        ),
    ),

    'pow_diff_dtype': dict( # llm used
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int32), Skip(np.int16)],
                },
            ]
        ),
    ),

    'bmm': dict( # llm used
        name=['bmm'],
        atol=3e-2,
        rtol=3e-2,
    ),

    'reduce_op': dict( # llm used
        name=['sum'],
        atol=1e-3,
        rtol=1e-3,
    ),

    'reduce_partial_op': dict( # llm used
        atol=1e-3,
        rtol=1e-3,
        name=['sum'],
    ),

    'mse_loss': dict(
        name=['mse_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32), Skip(np.float64)],
                    "shape": [Skip((16, 0)), Skip((4, 0, 9)),],
                },
            ]
        ),
    ),

    'cross_entropy': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3, 9)),Skip((64, 9)),Skip((5, 9, 12, 4))],
                },
            ]
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3, 5, 6, 6)),Skip((1024, 81)),Skip((64, 8, 8)),Skip((3, 5, 6, 6)),Skip((1024, 81)),Skip((64, 8))],
                },
            ]
        ),
    ),

    'linear': dict(
        name=['linear'],
        atol = 1e-1,
        rtol = 1e-1,
    ),

    'embedding': dict( # llm used
        name=["embedding"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["weight"],
                    # Wrong weight gradient. torch_npu failed in exactly the same way.
                    "shape": (Skip((93, 512)),),
                },
            ],
        ),
    ),

    'one_hot': dict(
        name=['one_hot'],
        para=dict(
            num_classes=[Skip(-1)],
        ),
    ),

    'split': dict( # llm used
        name=['split'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(np.float64)],
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
                    "shape": [Skip(())],
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
                    "dtype": [Skip(np.float64),],
                },
            ]
        ),
    ),

    'mm': dict( # llm used
        name=['mm'],
        atol=2e-2,
        rtol=2e-2,
    ),

    'mm_diff_dtype': dict( # llm used
        name=['mm'],
        atol=2e-2,
        rtol=2e-2,
    ),

    'permute': dict(
        name=['permute'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(())],
                    "dtype": [Skip(np.float64)],
                },
            ]
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        atol=5e-2,
        rtol=5e-2,
        atol_half=5e-2,
        rtol_half=5e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),],
                },
            ]
        ),
    ),

    'scatter_scalar': dict( # llm used
        name=['scatter'],
        para=dict(
            # In this case, for float32 (but not float64), no matter what the value parameter is,
            # the shape and dim parameters will result in wrong output for unknown reasons.
            # Specificially, the rows of elements that shouldn't get impacted by scatter,
            # will be filled with seemingly random or zero values.
            value=[Skip(1e-4),],
        ),
    ),

    'index_put_acc_three_indices': dict( # llm used
        name=['index_put'],
        para=dict(
            accumulate=[Skip(False)]
        ),
    ),

    'index_put_acc_two_indices': dict( # llm used
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 5)),],
                },
            ]
        ),
    ),

    'index_put_acc_one_indices': dict( # llm used
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((6,)), Skip((5,)), Skip((4, 5)),],
                },
            ]
        ),
    ),

    'index_put_bool_indices_value': dict( # llm used
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3, 2, 2, 20)),],
                },
            ]
        ),
    ),

    'layer_norm': dict(
        name=['layer_norm'],
        atol=1e-2,
        rtol=1e-3,
        para=dict(
            eps=[Skip(2),],
        ),
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'copy': dict( # llm used
        name=["copy_"],
        tensor_para=dict(
            # FIXME data type DT_COMPLEX128 of input [dst] is not supported
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((12, 0, 9)), Skip((8,))],
                    "dtype": [Skip(np.complex128), Skip(np.complex64), Skip(np.float64)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex128), Skip(np.float64)]
                },
            ]
        )
    ),

    'copy_input_no_contiguous': dict( # llm used
        name=["copy_"],
        tensor_para=dict(
            # FIXME not supported complex
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((12, 1, 12)),],
                    "dtype": [Skip(np.complex128), Skip(np.complex64), Skip(np.float64)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex64)]
                },
            ]
        )
    ),

    'copy_other_no_contiguous': dict( # llm used
        name=["copy_"],
        tensor_para=dict(
            # FIXME data type DT_COMPLEX64 of input [dst] is not supported
            # FIXME data type DT_COMPLEX128 of input [dst] is not supported
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((6, 5, 384)), Skip((2, 4, 38, 45))],
                    "dtype": [Skip(np.complex128), Skip(np.complex64), Skip(np.float64)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex128), Skip(np.float64)],
                },
            ]
        )
    ),

    'copy_all_no_contiguous': dict( # llm used
        name=["copy_"],
        tensor_para=dict(
            # FIXME data type DT_COMPLEX64 of input [dst] is not supported
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((192, 147)), Skip((192, 147, 2)), Skip((2, 12, 38, 45, 3))],
                    "dtype": [Skip(np.complex128), Skip(np.complex64), Skip(np.float64)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex64), Skip(np.float64)],
                },
            ]
        )
    ),

    'interpolate': dict(
        name=['interpolate'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 16, 23)),Skip((2, 256, 25, 38)),Skip((1, 3, 32, 224, 224)),Skip((2, 2, 16, 16)),Skip((2, 2, 16, 16)),Skip((2, 256, 13, 19)),Skip((3, 12, 14, 19)),Skip((2, 16, 1, 1)),Skip((2, 16, 15, 32)),Skip((1, 3, 32, 112, 112)),Skip((1, 3, 32, 112, 112)),Skip((2, 32, 32)),Skip((2, 32, 32)),Skip((2, 32, 32)),],
                },
            ]
        ),
    ),

    'repeat': dict( # llm used
        name=['repeat'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 2, 3, 5))],
                },
            ]
        ),
    ),

    'reduce_partial_op_4': dict( # llm used
        name=['sum'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-4,
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 2)), Skip((16, 0, 9))),
                },
            ],
        ),
    ),

    'remainder_self_scalar': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    # in case for zero division
    'remainder_self_bool': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    # in case for zero division
    'remainder_tensor': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    # in case for zero division
    'remainder_tensor_zero': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.uint8),Skip(np.int8),],
                },
            ]
        ),
    ),

    # in case for zero division
    'remainder_other_scalar': dict(
        name=['remainder'],
        para=dict(
            other=[Skip(0),],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8),],
                },
            ]
        ),
    ),

    # in case for zero division
    'remainder_other_scalar_bool': dict(
        name=['remainder'],
        para=dict(
            other=[Skip(False),],
        ),
    ),
}
