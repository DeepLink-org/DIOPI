# Copyright (c) 2023, DeepLink.
import numpy as np
from skip import Skip

device_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        # FIXME　slowfast等模型测例精度异常
        # atol=1e-2,
        # rtol=1e-3,
        atol=1e-1,
        rtol=1e-1,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # FIXME sar测例精度异常
                    "shape": [Skip((384, 64, 48, 160)), Skip((384, 128, 48, 160)),],
                    "dtype": [Skip(np.float16)]
                },
            ]
        ),
    ),

    'batch_norm_no_contiguous': dict(
        name=["batch_norm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.float32), Skip(np.float16), Skip(np.float64)]
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=["baddbmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((2, )),),
                },
                {
                    "ins": ["batch1"],
                    "shape": (Skip((2, 0, 4)),),
                },
                {
                    "ins": ["batch2"],
                    "shape": (Skip((2, 4, 2)),),
                },
            ]
        ),
    ),

    'hardtanh': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),

    'hardtanh_int': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),

    'hardtanh_uint': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),

    'threshold_uint': dict(
        name=["threshold"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8)],
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((16, 0)), Skip((4, 0, 9))),
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-3,
        rtol=1e-3,
        para=dict(
            # Now, there is a problem calculating for total weight,
            # which will be fixed in later cnnl kernel update.
            # See loss.cpp for more details
            reduction=[Skip('mean')],
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol_half=1e-1,
        rtol_half=1e-1,
        tensor_para=dict(
            args=[
                {
                    "ins": ["weight"],
                    "shape": [Skip((18, 8, 12, 2)), Skip((6, 9, 3, 5)), Skip((2048, 1, 3, 3)), Skip((2, 6, 2, 3))],
                },
            ]
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        # FIXME centernet精度异常
        atol=1e-1,
        rtol=1e-1,
        atol_half=1e-1,
        rtol_half=1e-1,
        tensor_para=dict(
            args=[
                {
                    # FIXME sar精度异常
                    "ins": ["weight"],
                    "shape": [Skip((128, 64, 3, 3))],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            # camb kernel only support dilation == 1
            dilation=[Skip((4, 3)), Skip((2, 3)), Skip(2)],
            # FIXME camb precision failed when return_indices is True
            return_indices=[Skip(True), Skip(True)],
        ),
        tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        # FIXME sar精度异常
                        "shape": [Skip((384, 128, 48, 160))],
                    },
                ]
        ),
    ),

    'max_pool2d_return_indices': dict(
        name=["max_pool2d"],
        para=dict(
            dilation=[Skip((4, 3)), Skip((2, 3))],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        atol=1e-2,
        rtol=1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((4, 0)), Skip((9, 0, 16))),
                    "dtype": [Skip(np.float16)],
                },
                {
                    "ins": ['weight'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64),
                              Skip(np.uint8), Skip(np.int8)],
                },
            ]
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        atol=1e-3,
        rtol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": (Skip((0,)), Skip((4, 0)), Skip((9, 0, 16))),
                },
                {
                    "ins": ['weight'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64),
                              Skip(np.uint8), Skip(np.int8)],
                },
                {
                    "ins": ['pos_weight'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64),
                              Skip(np.uint8), Skip(np.int8)],
                },
            ],
        ),
    ),

    # FIXME 模型测例精度异常
    'atan': dict(
        name=["atan"],
        atol=1e-3,
        rtol=1e-4,
    ),

    'pointwise_op': dict(
        name=['atan', 'erfinv'],
        atol=1e-3,
        rtol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ],
        ),
    ),

    'pointwise_op_int_without_inplace': dict(
        name=['atan'],
        atol=1e-3,
        rtol=1e-4,
    ),

    'pointwise_op_uint8': dict(
        name=['atan'],
        atol=1e-3,
        rtol=1e-4,
    ),

    'pointwise_binary': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # when dtype of input is uint8, output might overflow.
                    "dtype": [Skip(np.uint8)],
                },

            ],
        ),
    ),

    'pointwise_binary_diff_dtype_without_bool': dict(	
        name=['div'],	
        tensor_para=dict(	
            args=[	
                {	
                    "ins": ['input'],	
                    "dtype": [Skip(np.uint8)],	
                },	
                {	
                    "ins": ['other'],	
                    "dtype": [Skip(np.float16)],	
                },	
            ],	
        ),	
    ),

    'addmm': dict(
        name=["addmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                }
            ],
        ),
    ),

    'addcmul': dict(
        name=["addcmul"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64),
                              Skip(np.uint8), Skip(np.int8)],
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((5, 2, 16, 7)), Skip((3, 4, 16, 7))),
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((3, 16, 8)), Skip((4, 16, 12))),
                },
            ]
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        atol=1e-3,
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16), Skip(np.float64), Skip(np.float32)],
                },
            ],
        ),
    ),

    'clamp_max_tensor': dict(
        name=['clamp_max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16), Skip(np.float64), Skip(np.float32)],
                },
            ],
        ),
    ),

    'clamp_min_tensor': dict(
        name=['clamp_min'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16), Skip(np.float64), Skip(np.float32)],
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        atol=0.001,
        rtol=0.0001,
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        atol=0.001,
        rtol=0.0001,
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((12, 0)), Skip((9, 0, 7))),
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            # label_smoothing is not supported by camb kernel
            label_smoothing=[Skip(True), Skip(1), Skip(0.3), Skip(-1.3), Skip(0.4)],
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ],
        ),
    ),

    'select': dict(
        name=["select"],
        para=dict(
            # negative index can't get the correct result
            index=[Skip(-5)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((2, 0)), Skip((6, 0, 9))),
                },
            ]
        ),
    ),

    'select_not_float': dict(
        name=["select"],
        para=dict(
            # negative index can't get the correct result
            index=[Skip(-12)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((2, 0)), Skip((6, 0, 9))),
                },
            ]
        ),
    ),

    'index_select': dict(
        name=["index_select"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": (Skip((12, 0)), Skip((2, 0, 9))),
                },
            ]
        ),
    ),

    'index_select_not_float': dict(
        name=["index_select"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": (Skip((12, 0)), Skip((2, 0, 15))),
                },
            ]
        ),
    ),

    # not implement
    'masked_scatter': dict(
        name=["masked_scatter"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16),
                              Skip(np.int64), Skip(np.int32), Skip(np.int16),
                              Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
                },
            ],
        ),
    ),

    'masked_select': dict(
        name=['masked_select'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((4,)), Skip((4, 5, 6)), Skip(()), Skip((0,)), Skip((4, 0)), Skip((16, 0, 9))),
                },
                {
                    "ins": ['mask'],
                    "shape": (Skip((5, 6)), Skip((0,)), Skip((2, 4, 0)), Skip((0, 9))),
                },
            ],
        ),
    ),

    'masked_select_not_float': dict(
        name=['masked_select'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()), Skip((3, 4)), Skip((4, 6, 5, 8)), Skip((4, 1, 6)), Skip((0,)), Skip((4, 1)), Skip((16, 0, 9))),
                },
                {
                    "ins": ['mask'],
                    "shape": (Skip((2, 4)), Skip((5, 6)), Skip((0,)), Skip((4, 0)), Skip((1, 0, 9))),
                },
            ],
        ),
    ),

    'gather': dict(
        name=['gather'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((3, 9)), Skip((14, 6, 2)), Skip((2, 0)), Skip((5, 0, 9))),
                },
            ],
        ),
    ),

    'gather_0dim': dict(
        name=['gather'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['index'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'gather_not_float': dict(
        name=['gather'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((3, 9)), Skip((14, 6, 2)), Skip((2, 0)), Skip((5, 0, 9))),
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # when padding_idx set to 0, this case failed
                    "shape": [Skip((2, 3, 4))],
                },
            ],
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
            ],
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((0, 16, 20, 8)),),
                }
            ]
        ),
    ),

    'index_int': dict(
        name=["index"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64), Skip(np.int32), Skip(np.int16),
                              Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    'sgd_without_buf': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # not implement
    'max_pool3d': dict(
        name=['max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32), Skip(np.float16), Skip(np.float64)],
                },
            ]
        ),
    ),

    # not implement
    'max_pool3d_return_indices': dict(
        name=['max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32), Skip(np.float16), Skip(np.float64)],
                },
            ]
        ),
    ),

    # not implement
    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # not implement
    'adaptive_max_pool3d': dict(
        name=["adaptive_max_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # not implement
    'adaptive_max_pool3d_return_indices': dict(
        name=["adaptive_max_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    'adam': dict(
        name=['adam', 'adamw'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    # FIXME Run diopi_functions.adam failed, because of inputs: param_grad changed
                    # FIXME 特定参数组合精度差距过大
                    "shape": [Skip(()), Skip((512, 256, 3, 3)), Skip((512, 512, 3, 3))],
                },
            ]
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        para=dict(
            # Currently, p must be equal 1.0 due to the limitation of Cambrian operator.
            p=[Skip(2), Skip(0), Skip(0.5), Skip(float("inf")), Skip(1.2)],
        ),
    ),

    'cdist_compute_mode': dict(
        name=['cdist'],
        para=dict(
            # Currently, p must be equal 1.0 due to the limitation of Cambrian operator.
            p=[Skip(2)],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'adadelta': dict(
        name=["adadelta"],
        atol_half=1e-3,
        rtol_half=1e-3,
        tensor_para=dict(
            args=[
                {
                    # can't get correct result
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'mm': dict(
        name=['mm'],
        # FIXME 模型测例精度异常
        atol=1e-2,
        rtol=1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((8, 0)),),
                }
            ],
        ),
    ),

    'mm_diff_dtype': dict(
        name=['mm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((8, 0)),),
                }
            ],
        ),
    ),

    'expand': dict(
        name=['expand'],
        interface=['torch.Tensor'],
        para=dict(
            size=[Skip((0,))],
        ),
    ),

    'permute': dict(
        name=['permute'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(())],
                },
            ],
        ),
    ),

    'norm': dict(
        name=['norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ],
        ),
    ),

    'prod': dict(
        name=['prod'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ],
        ),
    ),

    # TODO: ctc_loss of camb could work correctly due to dipu and one_iter, need to fix diopi_test
    'ctc_loss': dict(
        name=["ctc_loss"],
        para=dict(
            blank=[Skip(0), Skip(9)]
        ),
    ),

    'ctc_loss_un_padded': dict(
        name=["ctc_loss"],
        para=dict(
            blank=[Skip(0), Skip(9)]
        ),
    ),

    'remainder_self_scalar': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.float16), Skip(np.int16), Skip(np.int32), Skip(np.int64), Skip(np.int8), Skip(np.uint8)],
                },
            ],
        ),
    ),

    'remainder_self_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64), Skip(np.int8), Skip(np.uint8)],
                },
            ],
        ),
    ),

    'remainder_tensor': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16), Skip(np.int8)],
                },
            ],
        ),
    ),

    'remainder_tensor_zero': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16), Skip(np.int8)],
                },
            ],
        ),
    ),

    'remainder_other_scalar': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16), Skip(np.int16), Skip(np.int32), Skip(np.int64), Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
                },
            ],
        ),
    ),

    'remainder_other_scalar_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 1))],
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        # FIXME sar精度异常
        atol=1e-02,
        rtol=1e-02,
        tensor_para=dict(
            args=[
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    # FIXME llama精度异常
                    "shape": (Skip((0, 8)), Skip((32000, 4096))),
                },
            ]
        ),
    ),

    'one_hot': dict(
        name=["one_hot"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'scatter_specific': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((5, 0)), Skip((4, 9, 0))),
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        para=dict(
            # The reduction operation of multiply is not supported by cnnl
            reduce=[Skip('multiply')],
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        para=dict(
            # The reduction operation of multiply is not supported by cnnl
            reduce=[Skip('multiply')],
        ),
    ),

    'index_put_acc_three_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                # index_put can't get the correct result
                {
                    "ins": ['input'],
                    "shape": [Skip((16, 4, 4)), Skip((4, 5, 0))],
                    "dtype": [Skip(np.uint8),    # overflow issue
                              Skip(np.bool_)],    # not supported by camb kernel when accumulate is true
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
                    "shape": [Skip((4, 5, 0))],
                },
                # index_put can't get the correct result
                {
                    "ins": ['indices1'],
                    "shape": [Skip((4, 5))],
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
                    "shape": [Skip((4, 5)), Skip((4, 0))],
                },
                # index_put can't get the correct result
                {
                    "ins": ['indices1'],
                    "shape": [Skip((6,)), Skip((2, 10)), Skip(())],
                },
            ]
        ),
    ),

    # when accumulate is True and dtype of indices is bool, can't get the correct result
    'index_put_acc_bool_indices_zeros': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['indices1'],
                    "dtype": [Skip(np.bool_)],
                },
            ]
        ),
    ),

    # when accumulate is True and dtype of indices is bool, can't get the correct result
    'index_put_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['indices1'],
                    "dtype": [Skip(np.bool_)],
                },
            ]
        ),
    ),

    # when accumulate is True and dtype of indices is bool, can't get the correct result
    'index_put_bool_indices_value': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['indices1'],
                    "dtype": [Skip(np.bool_)],
                },
            ]
        ),
    ),

    'unfold': dict(
        name=["unfold"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'pad': dict(
        name=['pad'],
        para=dict(
            # Only supports 2D padding for reflection/replicate padding mode now
            # pad should be greater than or equal to 0
            pad=[Skip((7, -14, 2, 3)), Skip((0, 1, -1, 3, 1, 2)), Skip((0, 2, -1, 1, 1, 5)),],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # input dims should be 4D for cnnlReflectionPad2d
                    "shape": [Skip((4, 5)),],
                },
            ],
        ),
    ),

    'constant_pad': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(())],
                },
            ],
        ),
    ),

    'unique': dict(
        name=['unique'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # when dtype is float64, can't get the correct result
                    "shape": (Skip((4, 64, 128)),),
                },
            ],
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)), Skip((4, 0)), Skip((3, 0, 9))],
                    "dtype": [Skip(np.float64), Skip(np.float16), Skip(np.int64), Skip(np.int32),
                              Skip(np.int16), Skip(np.int8)],
                },
            ],
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8), Skip(np.bool_)],
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)), Skip((4, 0)), Skip((5, 0, 7))],
                },
            ],
        ),
    ),

    'bernoulli_int': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64), Skip(np.int32), Skip(np.int16),
                              Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        # FIXME 模型测例精度异常
        # atol=1e-4,
        atol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((32,)), Skip((2, 16, 128))),
                },
            ]
        )
    ),

    'layer_norm_empty_tensor': dict(
        name=["layer_norm"],
        atol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((0,)), Skip((0, 12))),
                },
            ]
        )
    ),

    'interpolate': dict(
        name=["interpolate"],
        atol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # camb not supports 5d upsample nearest
                    # when shape is (2, 16, 23), can't get correct result
                    "shape": [Skip((2, 16, 23)), Skip((1, 3, 32, 224, 224)), Skip((4, 3, 64, 224, 224))],
                },
            ]
        )
    ),

    # not implement
    'cholesky': dict(
        name=['cholesky_ex'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32)],
                },
            ],
        ),
        requires_backward=[0],
        saved_args=dict(output=0),
    ),

    # not implement
    'triangular_solve': dict(
        name=['triangular_solve'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32)],
                },
            ],
        ),
        saved_args=dict(output=0),
    ),

    # not implement
    'normal_std_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # not implement
    'normal_mean_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # not implement
    'normal_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    'amax': dict(
        name=['amax'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'linalgqr': dict(
        name=['linalgqr'],
        atol=1e-4,
    ),

    'cast_dtype': dict(
        name=["cast_dtype"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.int32)]
                },
                {
                    "ins": ['out'],
                    "dtype": [Skip(np.float64), Skip(np.int32)]
                }
            ]
        ),
    ),

    'batch_norm_stats': dict(
        name=["batch_norm_stats"],
        atol=1e-2,
        rtol=5e-3,
    ),

    # not implement
    'rotary_emb': dict(
        name=["rotary_emb"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ],
        ),
    ),

    # not implement
    'rms_norm': dict(
        name=["rms_norm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32)],
                },
            ],
        ),
    ),

    'copy': dict(
        name=["copy_"],
        tensor_para=dict(
            # FIXME not supported complex
            # FIXME Can not cast from diopi_dtype_bool to diopi_dtype_uint8
            # FIXME Can not cast from diopi_dtype_int32 to diopi_dtype_float64
            # FIXME Can not cast from diopi_dtype_float16 to diopi_dtype_float64
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.complex64), Skip(np.complex128), Skip(np.float64)]
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex128), Skip(np.bool_)]
                },
            ]
        )
    ),

    'copy_input_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            # FIXME not supported complex
            # FIXME Can not cast from diopi_dtype_float64 to diopi_dtype_bool
            # FIXME Can not cast from diopi_dtype_float64 to diopi_dtype_int16
            # FIXME Can not cast from diopi_dtype_int64 to diopi_dtype_int16
            # FIXME Can not cast from diopi_dtype_bool to diopi_dtype_float64
            # FIXME Can not cast from diopi_dtype_int16 to diopi_dtype_float64
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.complex128), Skip(np.complex64), Skip(np.int16), Skip(np.bool_)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex64), Skip(np.int16), Skip(np.bool_)]
                },
            ]
        )
    ),

    'copy_other_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            # FIXME not supported complex
            # FIXME Can not cast from diopi_dtype_float64 to diopi_dtype_int32
            # FIXME Can not cast from diopi_dtype_int64 to diopi_dtype_uint8
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.complex128), Skip(np.complex64)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex128), Skip(np.float64), Skip(np.int64)],
                },
            ]
        )
    ),

    'copy_all_no_contiguous': dict(
        name=["copy_"],
        tensor_para=dict(
            # FIXME not supported complex
            # FIXME Can not cast from diopi_dtype_bool to diopi_dtype_int16
            # FIXME Can not cast from diopi_dtype_int16 to diopi_dtype_int8
            # FIXME Can not cast from diopi_dtype_float64 to diopi_dtype_float16
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.complex64), Skip(np.complex128), Skip(np.int16)],
                },
                {
                    "ins": ["other"],
                    "dtype": [Skip(np.complex64), Skip(np.int16), Skip(np.float64)],
                },
            ]
        )
    ),

    # FIXME input不为int，other为float时，精度异常
    'ge': dict(
        name=["ge"],
        para=dict(
            other=[Skip(0.5), Skip(0.6)],
        ),
    ),

    # FIXME input不为int，other为float时，精度异常
    'eq_case_2': dict(
        name=["eq"],
        para=dict(
            other=[Skip(100000000.0),],
        ),
    ),

    # FIXME input不为int，other为float时，精度异常
    'eq': dict(
        name=["eq"],
        para=dict(
            other=[Skip(100000000.0),],
        ),
    ),

    # FIXME input不为int，other为float时，精度异常
    'lt': dict(
        name=["lt"],
        para=dict(
            other=[Skip(0.111111), Skip(0.45), Skip(0.5)],
        ),
    ),

    # FIXME dyhead测例精度异常
    'im2col': dict(
        name=["im2col"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((1, 768, 32, 128)), Skip((1, 384, 64, 256))],
                },
            ],
        ),
    ),

    # FIXME llama测例精度异常
    'cumsum': dict(
        name=["cumsum"],
        interface=["torch"],
        atol=1e-2,
        rtol=1e-2
    ),

    # sar报错
    'mul': dict(
        name=["mul"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [Skip((384, 1, 512, 6, 40)), Skip((1, 1, 512, 6, 40))],
                },
            ],
        ),
    ),

    # FIXME llama报错
    'mul_2': dict(
        name=["mul"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(np.complex64)],
                },
            ],
        ),
    ),
}
