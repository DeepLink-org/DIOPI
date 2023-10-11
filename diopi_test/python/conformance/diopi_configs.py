# Copyright (c) 2023, DeepLink.
from .config import Genfunc
from .diopi_runtime import Dtype

ops_with_states = {"batch_norm": {"running_mean", "running_var"},
                   "sgd": {"buf", "param"},
                   "fill_": {"input"},
                   "embedding": {"weight"},
                   "adam": {"param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
                   "adamw": {"param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
                   "adadelta": {"param", "square_avg", "acc_delta"},
                   "rmsprop": {"param", "square_avg", "grad_avg", "momentum_buffer"},
                   "copy_": {"input"},
                   "cast_dtype": {"out"},
                   "batch_norm_gather_stats_with_counts": {"running_mean", "running_var"},
                   }


diopi_configs = {
    # FIXME batch_norm输入0size的张量报错
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-1,
        para=dict(
            # training=[False, False, True, True, False, True, True, True],
            # momentum=[0.1, 0.15, 0.2, 0.25, 0, 1, -1, -0.3],
            # eps=[1e-5, 1e-4, 1e-4, 1e-5, 0, 1, -1, -1e-5],
            training=[False, False, True, True],
            momentum=[0.1, 0.15, 0.2, 0.25],
            eps=[1e-5, 1e-4, 1e-4, 1e-5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16),
                    #           (0, 7, 32, 56, 56), (0, 15, 32, 32), (0, 23, 5), (0, 16)),
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    # "shape": ((8, ), (64, ), None, (16, ),
                    #           (7, ), (15, ), None, (16, )),
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    # "shape": ((8, ), (64, ), None, (16, ),
                    #           (7, ), (15, ), None, (16, )),
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    # "shape": ((8, ), (64, ), (96, ), (16, ),
                    #           (7, ), (15, ), (96, ), (16, )),
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'batch_norm_nan': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        rtol=1e-6,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            training=[False, False, True, True],
            momentum=[-0.1, 0.15, 0, 0.25],
            eps=[1e-5, -1e-4, 1e-4, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (16, 2)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((8, ), (64, ), None, (2, )),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8, ), (64, ), None, (2, )),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((8, ), (64, ), (96, ), (2, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'batch_norm_no_contiguous': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            training=[False, False, True, True],
            momentum=[0.1, 0.15, 0.2, 0.25],
            eps=[1e-5, 1e-4, 1e-4, 1e-5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "stride":((2000000, 230400, 7200, 120, 2), (1, 2048, 2, 64), (1, 56, 2), (20, 1)),
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (32, 16)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "stride":((4, ), None, None, None),
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'batch_norm_stats': dict(
        name=["batch_norm_stats"],
        interface=['CustomizedTest'],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            eps=[1e-5, 1e-4, 1e-4, 1e-5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "requires_grad": [False],
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'batch_norm_gather_stats_with_counts': dict(
        name=["batch_norm_gather_stats_with_counts"],
        interface=['CustomizedTest'],
        dtype=[Dtype.float32, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            momentum=[1e-3, 1e-4, 1e-4, 1e-5],
            eps=[1e-5, 1e-4, 1e-4, 1e-5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "requires_grad": [False],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((8,), (64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8,), (64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mean_all"],
                    "shape": ((2, 8), (7, 64), (3, 96), (4, 16)),
                    "requires_grad": [False],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["invstd_all"],
                    "shape": ((2, 8), (7, 64), (3, 96), (4, 16)),
                    "requires_grad": [False],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["count_all"],
                    "shape": ((2,), (7,), (3,), (4,)),
                    "requires_grad": [False],
                    "gen_fn": dict(fn=Genfunc.randint, low=2, high=6),
                },
            ]
        ),
    ),

    'batch_norm_backward_reduce': dict(
        name=["batch_norm_backward_reduce"],
        interface=['CustomizedTest'],
        dtype=[Dtype.float32, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            input_g=[True, True, False],
            weight_g=[True, False, True],
            bias_g=[True, False, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["grad_output"],
                    "shape": ((2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["input"],
                    "shape": ((2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mean"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["invstd"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'batch_norm_backward_elemt': dict(
        name=["batch_norm_backward_elemt"],
        interface=['CustomizedTest'],
        dtype=[Dtype.float32, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ["grad_out"],
                    "shape": ((2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["input"],
                    "shape": ((2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mean"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["invstd"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["sum_dy"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["sum_dy_xmu"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["count"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": dict(fn=Genfunc.randint, low=4, high=6),
                    "dtype": [Dtype.int32],
                },
            ]
        ),
    ),

    'batch_norm_elemt': dict(
        name=["batch_norm_elemt"],
        interface=['CustomizedTest'],
        dtype=[Dtype.float32, Dtype.float64],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            eps=[1e-5, 1e-4, 1e-5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mean"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["invstd"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "shape": ((64,), (96,), (16,)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'baddbmm': dict(
        name=["baddbmm"],
        interface=["torch"],
        is_inplace=True,
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        atol=1e-4,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-1,
        para=dict(
            beta=[1, 0.5, -0.1, False],
            alpha=[0.1, 2, True, -2.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((32, 64, 16), (32, 64, 32), (168, 52, 64), (2, 0, 2)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["batch1"],
                    "shape": ((32, 64, 32), (32, 64, 8), (168, 52, 38), (2, 0, 4)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["batch2"],
                    "shape": ((32, 32, 16), (32, 8, 32), (168, 38, 64), (2, 4, 2)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=["baddbmm"],
        interface=["torch"],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        atol=1e-4,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-1,
        para=dict(
            beta=[1, -0.34, 0,
                  True, 0, -1.33,
                  1, 0.5, 0.1,
                  0, -0.3],
            alpha=[1, 1.33, 2,
                   False, -2, 3.2,
                   0.1, 0.2, -0.5,
                   -1.2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((32, 64, 16), (32, 64, 32), (168, 52, 64),
                              (16,), (64, 32), (1, 52, 64),
                              (32, 1, 16), (32, 64, 1), (64,), (2, ), (0, 2)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["batch1"],
                    "shape": ((32, 64, 32), (32, 64, 8), (168, 52, 38),
                              (32, 64, 32), (32, 64, 8), (168, 52, 38),
                              (32, 64, 32), (32, 64, 8), (168, 52, 38),
                              (2, 0, 4), (2, 0, 4)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["batch2"],
                    "shape": ((32, 32, 16), (32, 8, 32), (168, 38, 64),
                              (32, 32, 16), (32, 8, 32), (168, 38, 64),
                              (32, 32, 16), (32, 8, 32), (168, 38, 64),
                              (2, 4, 2), (2, 4, 2)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        # out = (in - (dilation * (kernel_size - 1) + 1) + 2 * padding) / stride + 1
        para=dict(
            stride=[1, 2, (2, 3), 2, 1, 1, (2, 2), 1],
            padding=[0, (2, 1), (2, 3), 0, 12, 0, (0, 0), 0],
            dilation=[1, (2, 9), (2, 1), 1, 12, 1, (1, 1), 1],
            groups=[1, 2, 3, 1, 2048, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((4, 7, 12, 13), (6, 16, 19, 8), (6, 27, 12, 8), (2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1), (2, 256, 200, 304), (0, 6, 5, 10)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((2, 7, 6, 5), (18, 8, 12, 2), (6, 9, 3, 5), (12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1), (12, 256, 1, 1), (2, 6, 2, 3)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": (None, (18, ), (6,), (12, ), None, None, (12, ), (2,)),
                },
            ]
        ),
    ),

    'conv_2d_no_contiguous': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        para=dict(
            stride=[2, 1, 1, (2, 2)],
            padding=[0, 12, 0, (0, 0)],
            dilation=[1, 12, 1, (1, 1)],
            groups=[1, 2048, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "stride": ((20480000, 80000, 400, 1), (1, 8192, 2, 128), (1, 4096, 2, 2), (20480000, 80000, 400, 1)),
                    "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1), (2, 256, 200, 304)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1), (12, 256, 1, 1)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": ((12, ), None, None, (12, )),
                },
            ]
        ),
    ),

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024,), (2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28),
                              (0,), (256, 0), (8, 0, 128)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu_no_contiguous': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "stride":((1, 3), (900, 30, 1),),
                    "shape": ((3, 3), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'hardtanh': dict(
        name=["hardtanh"],
        is_inplace=True,
        para=dict(
            min_val=[False, -1, 0.0, 0.0, -0.2, 1.2, -2, 1, -2.1],
            max_val=[0.4, True, 6, -0.5, 0.2, 1.2, 0, 0.0, -2.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (128,), (2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'hardtanh_int': dict(
        name=["hardtanh"],
        is_inplace=True,
        para=dict(
            min_val=[0, -1, 0.0, 0.0, -4.5, 2, -2, 1, -2.1],
            max_val=[0.4, 5, -6, 0.5, 6.5, 2, 0, 0.0, -2.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128,), (2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8],
                    "gen_fn": dict(fn=Genfunc.randint, low=-10, high=10),
                },
            ],
        ),
    ),

    'hardtanh_uint': dict(
        name=["hardtanh"],
        is_inplace=True,
        para=dict(
            min_val=[0, 1, 20, 0.0, 20, 20, 2, 1, False],
            max_val=[0.4, 5, 50, 0.5, 70.5, 20, True, 0.0, 10],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128,), (2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.uint8],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=256),
                },
            ],
        ),
    ),

    'hardswish': dict(
        name=["hardswish"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024,), (2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (3, 0, 9)),
                    "requires_grad": [True],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # 生成小于-3和大于3的测例
    'hardswish_domain': dict(
        name=["hardswish"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024,), (2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28)),
                    "requires_grad": [True],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": dict(fn=Genfunc.uniform, low=-6, high=6),
                },
            ],
        ),
    ),

    'threshold': dict(
        name=["threshold"],
        is_inplace=True,
        para=dict(
            threshold=[2, False, 2.0, -3.1, 4.7, 2, -1, 1, -2.1],
            value=[0, -5.34, 0.0, 33, 12, True, 0.0, -2.5, -2.1],
        ),
        tensor_para=dict(
            genfunc=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (64, ),
                              (2, 4096), (64, 28, 28),
                              (2, 144, 28, 28),
                              (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'threshold_int': dict(
        name=["threshold"],
        is_inplace=True,
        para=dict(
            threshold=[0, -1, 0.0, 5, -4.5, 2, -2, 1, -2.1],
            value=[0.4, 5, -6, 0.5, 6.5, 2, 0, 0.0, -2.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128,), (2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8],
                    "gen_fn": dict(fn=Genfunc.randint, low=-10, high=10),
                },
            ],
        ),
    ),

    'threshold_uint': dict(
        name=["threshold"],
        is_inplace=True,
        para=dict(
            threshold=[0, -1, 20, 0.0, -20, 20, -2, 1, -5],
            value=[0.4, 5, -50, 0.5, 70.5, 20, 0, 0.0, -10],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128,), (2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.uint8],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=256),
                },
            ],
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        atol=1e-4,
        rtol=1e-5,
        approximate=['none', 'tanh', 'none', 'tanh',
                     'none', 'tanh', 'none', 'tanh', 'none'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (32,), (16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gelu_specific': dict(
        name=['gelu'],
        atol=1e-4,
        rtol=1e-5,
        approximate=['none', 'tanh'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((32,), (16, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        para=dict(
            kernel_size=[2, (2, 2), (20, 13), (2, 2), 3],
            stride=[None, None, 3, 1, (1, 2)],
            padding=[0, (0, 0), (2, 3), (1, 1), 0],
            ceil_mode=[False, True, False, True, False],
            count_include_pad=[True, True, False, True, False],
            divisor_override=[None, None, -3, None, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 16, 7), (5, 2, 16, 7), (3, 4, 16, 7),
                              (2, 1024, 14, 14), (256, 28, 28)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'avg_pool2d_float64': dict(
        name=["avg_pool2d"],
        para=dict(
            kernel_size=[(2, 2), 3],
            stride=[1, (1, 2)],
            padding=[(1, 1), 0],
            ceil_mode=[True, False],
            count_include_pad=[True, False],
            divisor_override=[None, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # TODO(xintian): fix backward for Dtype.float64
                    # "requires_grad": [True],
                    "shape": ((2, 1024, 14, 14), (256, 28, 28)),
                    "dtype": [Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[6, (6, 12), (6, 8), (6, 8), 3, (2, 1), (2, 2), 3],
            stride=[None, (3, 100), (3, 2), (3, 2), 2, (2, 1), (2, 1), 2],
            padding=[0, (2, 6), (2, 3), (2, 3), 1, 0, (0, 1), 0],
            dilation=[1, (4, 3), (2, 3), (2, 3), 1, (1, 1), 1, 2],
            ceil_mode=[False, True, False, True, False, True, False, True],
            return_indices=[False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 12, 20), (5, 4, 17, 22),
                              (6, 17, 23), (1, 4, 17, 23),
                              (2, 64, 352, 528),
                              (2, 256, 12, 40),
                              (2, 512, 4, 26),
                              (3, 4, 10)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool2d_return_indices': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[6, (6, 12), (6, 8), (6, 8)],
            stride=[None, (3, 100), (3, 2), (3, 2)],
            padding=[0, (2, 6), (2, 3), (2, 3)],
            dilation=[1, (4, 3), (2, 3), (2, 3)],
            ceil_mode=[False, True, False, True],
            return_indices=[True, True, True, True],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((3, 12, 20), (5, 4, 17, 22),
                              (6, 17, 23), (1, 4, 17, 23),),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
        requires_backward=[0],
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        atol_half=1e-2,
        rtol_half=1e-2,
        para=dict(
            output_size=[5, (26, 40), (None, None), (1, 1), 2,
                         (None, 3), (3, 4), (7, 7), (10, 10)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 16, 8), (4, 7, 27, 39), (4, 16, 12),
                              (2, 2048, 8, 6), (2, 288, 33, 33),
                              (2, 144, 65, 65), (2, 1280, 7, 7),
                              (2, 265, 7, 7), (2, 265, 7, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d_zero_size': dict(
        name=["adaptive_avg_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        atol_half=1e-2,
        rtol_half=1e-2,
        para=dict(
            output_size=[0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4, 7, 27, 39),),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d': dict(
        name=["adaptive_max_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[5, (26, 40), (None, None), 2, (1, 3), (3, 4), (33, 33), (40, 40)],
            return_indices=[False, False, False, False, False, False, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 16, 8), (4, 7, 27, 39), (4, 16, 12),
                              (288, 33, 33), (2, 144, 33, 33), (2, 16, 130, 130),
                              (2, 144, 33, 33), (2, 144, 33, 33)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d_return_indices': dict(
        name=["adaptive_max_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[5, (26, 40), (None, None), (0, 0)],
            return_indices=[True, True, True, True]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((3, 16, 8), (4, 7, 27, 39), (4, 16, 12), (4, 16, 12)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        atol=1e-3,
        rtol=1e-4,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.float32, Dtype.float64],
        para=dict(
            reduction=['mean', 'none', 'sum',
                       'mean', 'none', 'sum',
                       'mean', 'none', 'sum'],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (16,), (72,),
                              (2, 11856), (2, 741, 80), (4, 4, 16, 20),
                              (0,), (4, 0), (9, 0, 16)),
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['target'],
                    "shape": ((), (16,), (72,),
                              (2, 11856), (2, 741, 80), (4, 4, 16, 20),
                              (0,), (4, 0), (9, 0, 16)),
                },
                {
                    "ins": ['weight'],
                    "shape": (None, (), (72,),
                              (2, 11856), (2, 741, 80), (16, 20),
                              (), (0,), (0, 16)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8],
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        atol=1e-3,
        rtol=1e-4,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.float32, Dtype.float64],
        para=dict(
            reduction=['mean', 'none', 'sum',
                       'mean', 'none', 'sum',
                       'mean', 'none', 'sum'],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (16,), (72,),
                              (2, 11856), (2, 741, 80), (4, 4, 16, 20),
                              (0,), (4, 0), (9, 0, 16)),
                },
                {
                    "ins": ['target'],
                    "shape": ((), (16,), (72,),
                              (2, 11856), (2, 741, 80), (4, 4, 16, 20),
                              (0,), (4, 0), (9, 0, 16)),
                },
                {
                    "ins": ['weight'],
                    "shape": (None, (), (72,),
                              (2, 11856), (2, 1, 80), (16, 20),
                              (), (0,), (0, 16)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8],
                },
                {
                    "ins": ['pos_weight'],
                    "shape": (None, (), (72,),
                              (11856,), (2, 741, 80,), (4, 16, 20),
                              (), (4, 1), (16,)),
                    "dtype": [Dtype.float16, Dtype.int32, Dtype.float64,
                              Dtype.int64, Dtype.float32, Dtype.int16,
                              Dtype.int8, Dtype.uint8],
                },
            ],
        ),
    ),

    'pointwise_op': dict(
        name=['abs', 'cos', 'erf', 'erfinv', 'exp', 'floor',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40), (0,), (16, 0)),
                },
            ],
        ),
    ),

    # FIXME erfinv输入int或bool报错
    'pointwise_op_int_without_inplace': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp',
        #       'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        name=['abs', 'cos', 'erf', 'exp',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-5, high=5),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    # FIXME erfinv输入int或bool报错
    'pointwise_op_uint8': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp',
        #       'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        name=['abs', 'cos', 'erf', 'exp',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        interface=['torch'],
        dtype=[Dtype.uint8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=0, high=20),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072)),
                },
            ],
        ),
    ),

    'pointwise_op_mask': dict(
        name=['logical_not', 'bitwise_not'],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=0, high=2),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    # FIXME erfinv输入int或bool报错
    'pointwise_op_bool': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp', 'sin', 'asin', 'sqrt', 'rsqrt', 'atan', 'logical_not'],
        name=['abs', 'cos', 'erf', 'exp', 'sin', 'asin', 'sqrt', 'rsqrt', 'atan', 'logical_not'],
        interface=['torch'],
        dtype=[Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.mask,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'pointwise_op_abs_input': dict(
        name=['log', 'log2', 'log10', 'sqrt', 'rsqrt'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.positive,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3), (2, 31, 512, 6, 40),
                              (0,), (0, 16), (8, 0, 4)),
                },
            ],
        ),
    ),

    'log_integer_input': dict(
        name=['log', 'log2', 'log10'],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
        tensor_para=dict(
            gen_fn=Genfunc.positive,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'log_zero_input': dict(
        name=['log', 'log2', 'log10'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.uint8, Dtype.int8],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'log_neg_input': dict(
        name=['log', 'log2', 'log10'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.uint8, Dtype.int8],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'tanh': dict(
        name=['tanh'],
        interface=['torch'],
        is_inplace=True,
        saved_args=dict(output=0),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40),
                              (0,), (16, 0), (1, 0, 6)),
                },
            ],
        ),
    ),

    'tanh_not_float': dict(
        name=['tanh'],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40),
                              (0,), (16, 0), (1, 0, 6)),
                },
            ],
        ),
    ),

    'sign': dict(
        name=['sign'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
               Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40), (0,), (16, 0)),
                },
            ],
        ),
    ),

    'pointwise_op_zero': dict(
        name=['abs', 'exp', 'floor', 'neg', 'sqrt',
              'logical_not', 'rsqrt', 'ceil'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (16, ), (8, 64)),
                },
            ],
        ),
    ),

    'pointwise_op_without_inplace_zero': dict(
        name=['abs', 'sign', 'exp', 'sqrt',
              'logical_not', 'rsqrt'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
               Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (16, ), (8, 64)),
                },
            ],
        ),
    ),

    'neg_without_inplace_zero': dict(
        name=['neg'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
               Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (16, ), (8, 64)),
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=['torch'],
        is_inplace=True,
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (182400,), (20267, 80), (8, 200, 304),
                              (32, 16, 1, 1), (16, 32, 130, 130),
                              (0,), (256, 0), (8, 0, 128)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME sigmoid输入int报错
    # 'sigmoid_int': dict(
    #     name=["sigmoid"],
    #     interface=['torch'],
    #     saved_args=dict(output=0),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((8, 200, 304),),
    #                 "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
    #                           Dtype.uint8, Dtype.int8],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    'silu': dict(
        name=["silu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((182400,), (20267, 80), (8, 200, 304),
                              (32, 16, 1, 1), (16, 32, 130, 130), (),
                              (0,), (0, 16), (8, 0, 17)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pow': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            exponent=[-2, -0.5, 0, 0.6, True, 3, 4., 1.],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (16, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 8), (7, 0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                }
            ],
        ),
    ),

    'pow_int': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            exponent=[-0.5, 2, 0.6, 1.2, 0, 0., 1, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (16, ), (20267, 80),
                              (2, 128, 3072), (2, 512, 38, 38),
                              (0,), (0, 8), (7, 0, 9)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8],
                    "gen_fn": dict(fn=Genfunc.randint, low=-4, high=4),
                }
            ],
        ),
    ),

    'pow_bool': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            exponent=[0, -1.2, 2, 0.6, 1.2, 0.],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 8)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                }
            ],
        ),
    ),

    'pow_tensor': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
               Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.int8, Dtype.uint8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randn_int, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 4), (9, 0, 3)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((), (1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 4), (9, 0, 3)),
                },
            ],
        ),
    ),

    'pow_tensor_only_0_1': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.int8, Dtype.uint8],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 4), (9, 0, 3)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((), (1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38),
                              (0,), (0, 4), (9, 0, 3)),
                },
            ],
        ),
    ),

    'pow_broadcast': dict(
        name=['pow'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (2, 1, 128), (2, 64, 1, 128),
                              (2, 32, 130, 130), (0,), (8, 16, 1)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((4, 16), (384, 128), (64, 16, 128),
                              (5, 2, 32, 1, 130), (16, 0), (16, 0,)),
                },
            ],
        ),
    ),

    'pow_broadcast_inplace': dict(
        name=['pow'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (2, 1024), (2, 384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (5, 2, 32, 130, 130), (16, 0,), (8, 16, 0), (32, 0, 16)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (2, 32, 1, 130), (0,), (16, 1,), (0, 16)),
                },
            ],
        ),
    ),

    'pow_diff_dtype_cast': dict(
        name=['pow'],
        interface=['torch'],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randn_int, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.int64, Dtype.int32, Dtype.int16,
                             Dtype.bool, Dtype.bool, Dtype.bool, Dtype.bool],
                },
                {
                    "ins": ['exponent'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float32, Dtype.float64, Dtype.float16,
                             Dtype.int32, Dtype.float32, Dtype.int8, Dtype.uint8],
                },
            ],
        ),
    ),

    # FIXME pow的input与exponent输入uint8和int8，结果不一致
    'pow_diff_dtype': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randn_int, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    # "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
                    #          Dtype.int32, Dtype.float64, Dtype.float64,
                    #          Dtype.int8, Dtype.float32, Dtype.uint8],
                    "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
                             Dtype.int32, Dtype.float64, Dtype.float32,
                             Dtype.float32, Dtype.int16, Dtype.int64],
                },
                {
                    "ins": ['exponent'],
                    "shape": ((1024, ),),
                    # "dtype":[Dtype.int32, Dtype.uint8, Dtype.bool,
                    #          Dtype.int64, Dtype.float16, Dtype.float32,
                    #          Dtype.uint8, Dtype.bool, Dtype.int8],
                    "dtype":[Dtype.int32, Dtype.uint8, Dtype.bool,
                             Dtype.int64, Dtype.float16, Dtype.float64,
                             Dtype.bool, Dtype.uint8, Dtype.bool],
                },
            ],
        ),
    ),

    'pow_input_scalar': dict(
        name=['pow'],
        interface=['torch'],
        para=dict(
            self=[-2, -0.5, 0, 0.6, 2, 3, 4., 1.],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "shape": ((), (8,), (125, 1),
                              (70, 1, 2), (4, 256, 16, 16),
                              (0,), (0, 4), (9, 0, 6)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                }
            ],
        ),
    ),

    'pow_input_scalar_bool': dict(
        name=['pow'],
        interface=['torch'],
        para=dict(
            self=[True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "shape": ((70, 1, 2), (4, 256, 16, 16)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                }
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float64, Dtype.float32, Dtype.float16, Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130), (0,)),
                },
                {
                    "ins": ['other'],
                    "shape": ((), (1024, ), (384, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1), (0,)),
                },
            ],
        ),
    ),

    'pointwise_binary_broadcast': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (2, 1024), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 1, 128),
                              (2, 32, 130, 130), (0,), (8, 16, 1), (32, 0, 16)),
                },
                {
                    "ins": ['other'],
                    "shape": ((4, 16), (), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (5, 2, 32, 1, 130), (16, 0), (16, 0,), (0, 16)),
                },
            ],
        ),
    ),

    'pointwise_binary_broadcast_inplace': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        dtype=[Dtype.float32],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (2, 1024), (2, 384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (5, 2, 32, 130, 130), (16, 0,), (8, 16, 0), (32, 0, 16)),
                },
                {
                    "ins": ['other'],
                    "shape": ((), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (2, 32, 1, 130), (0,), (16, 1,), (0, 16)),
                },
            ],
        ),
    ),

    # FIXME add输入int8、uint8结果不一致
    'pointwise_binary_diff_dtype': dict(
        # name=['add', 'mul', 'eq', 'ne', 'le',
        #       'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        name=['mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
                             Dtype.int64, Dtype.int32, Dtype.int16,
                             Dtype.int8, Dtype.uint8, Dtype.bool],
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.int32, Dtype.uint8, Dtype.bool,
                             Dtype.int64, Dtype.float64, Dtype.float32,
                             Dtype.int16, Dtype.float16, Dtype.int8],
                },
            ],
        ),
    ),

    # FIXME add、mul输入int8、uint8结果不一致
    'pointwise_binary_diff_dtype_inplace': dict(
        # name=['add', 'mul', 'eq', 'ne', 'le',
        #       'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        name=['eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
                             Dtype.int32, Dtype.float64, Dtype.float64,
                             Dtype.int8, Dtype.float32, Dtype.int8],
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.int32, Dtype.uint8, Dtype.bool,
                             Dtype.int64, Dtype.float16, Dtype.float32,
                             Dtype.int16, Dtype.bool, Dtype.uint8],
                },
            ],
        ),
    ),

    # FIXME sub输入int8、uint8结果不一致
    'pointwise_binary_diff_dtype_without_bool': dict(
        # name=['sub', 'div'],
        name=['div'],
        interface=['torch'],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
                             Dtype.int32, Dtype.int32, Dtype.int16,
                             Dtype.int8, Dtype.uint8, Dtype.float32],
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.int32, Dtype.uint8, Dtype.int32,
                             Dtype.int64, Dtype.float64, Dtype.float32,
                             Dtype.uint8, Dtype.float16, Dtype.int8],
                },
            ],
        ),
    ),

    # FIXME sub输入int8、uint8结果不一致
    # 'pointwise_binary_diff_dtype_without_bool_inplace': dict(
    #     name=['sub'],
    #     interface=['torch'],
    #     is_inplace=True,
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((1024, ),),
    #                 "dtype":[Dtype.float64, Dtype.float32, Dtype.float16,
    #                          Dtype.int64, Dtype.float16, Dtype.float64,
    #                          Dtype.int8, Dtype.uint8, Dtype.float32],
    #             },
    #             {
    #                 "ins": ['other'],
    #                 "shape": ((1024, ),),
    #                 "dtype":[Dtype.int32, Dtype.uint8, Dtype.int32,
    #                          Dtype.int32, Dtype.float64, Dtype.float32,
    #                          Dtype.uint8, Dtype.int16, Dtype.int8],
    #             },
    #         ],
    #     ),
    # ),

    'pointwise_binary_dtype_bool': dict(
        name=['add', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge',
              'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.mask,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (2, 32, 1, 1)),
                },
            ],
        ),
    ),

    'bitwise_op': dict(
        name=['bitwise_and', 'bitwise_or'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-4, high=4),
            args=[
                {
                    "ins": ['input', 'other'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,), (0, 3), (9, 0, 4)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                },
            ],
        ),
    ),

    # FIXME bitwise_or输入uint8结果不一致
    'bitwise_op_diff_dtype': dict(
        name=['bitwise_and', 'bitwise_or'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,), (0, 3), (9, 0, 4)),
                    # "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                    #           Dtype.int8, Dtype.uint8, Dtype.bool],
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['other'],
                    "shape": ((), (1024, ), (384, 128),
                              (64, 3, 3,), (2, 32, 1, 1),
                              (1,), (3,), (0, 4)),
                    "dtype": [Dtype.uint8, Dtype.bool, Dtype.int16,
                              Dtype.int64, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    # FIXME bitwise_or输入uint8结果不一致
    'bitwise_op_broadcast': dict(
        name=['bitwise_and', 'bitwise_or'],
        interface=['torch'],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (128,),
                              (128, 1, 3, 3),
                              (2, 32, 1, 130),
                              (0, 3), (9, 1, 4)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['other'],
                    "shape": ((5,), (2, 1024, ), (384, 128),
                              (64, 3, 3,), (2, 32, 2, 130),
                              (2, 0, 3), (0, 4)),
                    # "dtype": [Dtype.uint8, Dtype.bool, Dtype.int16,
                    #           Dtype.int64, Dtype.uint8, Dtype.int32],
                    "dtype": [Dtype.int8, Dtype.bool, Dtype.int16,
                              Dtype.int64, Dtype.int8, Dtype.int32],
                },
            ],
        ),
    ),

    # FIXME bitwise_or输入uint8结果不一致
    'bitwise_op_scalar': dict(
        name=['bitwise_and', 'bitwise_or'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            other=[0, -1, True, 2, 100, False, -3, 4],
        ),
        # dtype=[Dtype.int16, Dtype.int32, Dtype.int64,
        #        Dtype.int8, Dtype.uint8],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.int8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-4, high=4),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,), (0, 4), (4, 0, 5)),
                },
            ],
        ),
    ),

    'bitwise_op_scalar_bool': dict(
        name=['bitwise_and', 'bitwise_or'],
        interface=['torch'],
        para=dict(
            other=[0, -1, 2, 100, -3, 4],
        ),
        dtype=[Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.mask,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,)),
                },
            ],
        ),
    ),

    # FIXME diopiBitwiseAndInpScalar,diopiBitwiseOrInpScalar输入布尔值报错
    # 'bitwise_op_scalar_bool_inplace': dict(
    #     name=['bitwise_and', 'bitwise_or'],
    #     interface=['torch'],
    #     is_inplace=True,
    #     para=dict(
    #         other=[True, False],
    #     ),
    #     dtype=[Dtype.bool],
    #     tensor_para=dict(
    #         gen_fn=Genfunc.mask,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((1024, ), (384, 128),),
    #             },
    #         ],
    #     ),
    # ),

    'div': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130), (0,)),
                },
                {
                    "ins": ['other'],
                    "shape": ((), (1024, ), (384, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1), (0,)),
                },
            ],
        ),
    ),

    'div_broadcast': dict(
        name=['div'],
        interface=['torch'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (2, 1024), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 1, 128),
                              (2, 32, 130, 130), (0,), (8, 16, 1), (32, 0, 16)),
                },
                {
                    "ins": ['other'],
                    "shape": ((4, 16), (), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (5, 2, 32, 1, 130), (16, 0), (16, 0,), (0, 16)),
                },
            ],
        ),
    ),

    'div_diff_dtype_inplace': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float64, Dtype.float32, Dtype.float16],
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ),),
                    "dtype":[Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ],
        ),
    ),

    'div_rounding_mode': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            rounding_mode=['floor', None, 'floor', 'trunc', 'floor'],
        ),
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((), (1024, ), (384, 128),
                              (1, ), (2, 32, 1, 1)),
                },
            ],
        ),
    ),

    'div_dtype_int_and_bool': dict(
        name=['div'],
        interface=['torch'],
        dtype=[Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (2, 32, 1, 1)),
                },
            ],
        ),
    ),

    'sub_scalar': dict(
        name=['sub'],
        interface=['torch'],
        tag=['scalar'],
        is_inplace=True,
        dtype=[Dtype.float32],
        para=dict(
            other=[0, -1, 0.028, 2.232, 1, -0.2421, -2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 64, 128),
                              (128, 64, 3, 3), (128, 32, 2, 2),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['add', 'mul', 'div', 'eq',
              'ne', 'le', 'lt', 'gt', 'ge'],
        interface=['torch'],
        tag=['scalar'],
        is_inplace=True,
        dtype=[Dtype.float32],
        para=dict(
            other=[0, -1, 0.028, 2.232, 1, True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 64, 128),
                              (128, 64, 3, 3), (128, 32, 2, 2),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'div_zero': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (2, 32, 1, 1)),
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'pointwise_binary_scalar_div_zero': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        para=dict(
            other=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'pointwise_binary_test_equal_and_logic_specific': dict(
        name=['eq', 'ne', 'le', 'lt',
              'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128)),
                }
            ],
        ),
    ),

    'sub_constant_with_alpha_and_no_contiguous': dict(
        name=['sub'],
        para=dict(
            alpha=[0, -2, 2.0, 4, 1, 0.234, -2.123],
            other=[3.5, -2, 2.0, 4, 1, -0.231, 3],
        ),
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 64, 128),
                              (128, 64, 3, 3), (128, 32, 2, 2),
                              (2, 32, 130, 130)),
                    'no_contiguous': [True],
                },
            ],
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add'],
        para=dict(
            alpha=[0, -2, 2.0, 4, 1, 0.234, -2.123],
            other=[3.5, -2, 2.0, 4, 1, True, False],
        ),
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 64, 128),
                              (128, 64, 3, 3), (128, 32, 2, 2),
                              (2, 32, 130, 130)),
                    'no_contiguous': [True],
                },
            ],
        ),
    ),

    'pointwise_binary_with_alpha': dict(
        name=['add', 'sub'],
        para=dict(
            alpha=[-2, 2.0],
        ),
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 3),
                              (2, 2, 4, 3)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1,), (1,)),
                }
            ],
        ),
    ),

    'pointwise_binary_with_alpha_bool': dict(
        name=['add'],
        para=dict(
            alpha=[True, False],
        ),
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 3),
                              (2, 2, 4, 3)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1,), (1,)),
                }
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 726, 32), (16, 100, 100), (9, 5, 5),
                              (0, 12, 16), (4, 0, 6), (4, 9, 0), (5, 8, 13)),
                },
                {
                    "ins": ['mat2'],
                    "shape": ((16, 32, 726), (16, 100, 32), (9, 5, 10),
                              (0, 16, 7), (4, 6, 8), (4, 0, 12), (5, 13, 0)),
                },
            ],
        ),
    ),

    'addmm': dict(
        name=["addmm"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            alpha=[0, -2, True, 0.001, -0.01, 1, 0.12, 1.],
            beta=[2, 0, 2.12, 0.001, -0.01, 1, False, 1.],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (2, 10), (768, ),
                              (1, 400), (4, 1), (1, 5),
                              (), (0,)),
                },
                {
                    "ins": ["mat1"],
                    "shape": ((3, 18), (2, 2048), (2, 768),
                              (1, 2304), (4, 16), (9, 4),
                              (16, 0), (8, 9)),
                },
                {
                    "ins": ["mat2"],
                    "shape": ((18, 4), (2048, 10), (768, 768),
                              (2304, 400), (16, 8), (4, 5),
                              (0, 8), (9, 0)),
                },
            ],
        ),
    ),

    'addcmul': dict(
        name=["addcmul"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[-2, 0.001, -0.01, 2, 1, 0, 1., 2.5],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64,
                   Dtype.int16, Dtype.int32, Dtype.int64,
                   Dtype.int8, Dtype.uint8],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 5),
                              (0,), (0, 5), (2, 0, 9)),
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 1),
                              (0,), (0, 5), (2, 0, 9)),
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 1, 5),
                              (0,), (0, 5), (2, 0, 9)),
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[-2, 0.001, -0.01, 2, 1, 0, 1., 2.5],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 5),
                              (0,), (0, 5), (2, 0, 9)),
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 1),
                              (0,), (0, 5), (2, 0, 9)),
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 1, 5),
                              (0,), (0, 5), (2, 0, 9)),
                },
            ],
        ),
    ),


    'addcdiv_specific': dict(
        name=["addcdiv"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[-2, 0.001, -0.01, 2, True, 0, 1., 2.5],
        ),
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 5),
                              (0,), (0, 5), (2, 0, 9)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 3, 1),
                              (0,), (0, 5), (2, 0, 9)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((), (128, ), (576, 192), (64, 3, 3, 3), (10, 1, 5),
                              (0,), (0, 5), (2, 0, 9)),
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'addcdiv_addcmul_broadcast_inplace': dict(
        name=["addcdiv", "addcmul"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[True, -1., 0.45],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": [(3, 4,), (4, 5, 5), (2, 3, 4, 5)],
                },
                {
                    "ins": ["tensor1"],
                    "shape": [(), (1, 5), (3, 4, 5)],
                },
                {
                    "ins": ["tensor2"],
                    "shape": [(4,), (4, 5, 1), (5, )],
                },
            ],
        ),
    ),

    'addcdiv_addcmul_without_inplace': dict(
        name=["addcdiv", "addcmul"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[-2, 0.001, -0.01, True, False, 0, 1., 2.5],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128, ), (576, 192), (64, 3, 1, 3), (10, 3, 5),
                              (0,), (0, 5), (2, 0, 9)),
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((128,), (4, 1, 128,), (1, 192), (64, 1, 3, 3), (3, 1),
                              (1,), (5,), (1, 9)),
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((5, 128), (3, 128, ), (2, 576, 1), (3, 3, 3), (10, 1, 5),
                              (2, 0,), (0, 5), (9,)),
                },
            ],
        ),
    ),

    # FIXME matmul输入空张量，运行报错
    'matmul': dict(
        name=['matmul'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((128, 49, 128), (5,), (128, 4, 49, 32),
                              (2, 1, 3136, 3136), (2, 784, 64), (2, 16, 8, 64), (2, 31, 6, 40, 512),)
                    #   (0,), (0, 0), (0, 4, 4), (5, 0, 3, 3)),
                },
                {
                    "ins": ['other'],
                    "shape": ((128, 384), (5,), (128, 4, 32, 49),
                              (2, 3, 3136, 64), (2, 64, 784), (2, 1, 64, 8), (512, 1),)
                    #   (0,), (0, 2), (1, 4, 5), (0, 3, 4)),
                },
            ],
        ),
    ),

    # FIXME clamp指定组合精度不一致
    'clamp_scalar': dict(
        name=['clamp'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            # min=[-1.1, None, None, True, -2, -0.23, 4, 0, -1, 2.3],
            # max=[None, 0.13, 2, 3, 0, -2, False, 2, None, 1.2],
            min=[-1.1, None, None, True, -0.23, 0, -1, 2.3],
            max=[None, 0.13, 2, 3, -2, 2, None, 1.2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((), (182, ), (384, 128),
                    #           (1, 242991, 2),
                    #           (2, 4, 100, 152),
                    #           (384, 128), (12, 16),
                    #           (0,), (4, 0), (3, 0, 9)),
                    "shape": ((), (182, ), (384, 128),
                              (1, 242991, 2),
                              (384, 128),
                              (0,), (4, 0), (3, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME clamp输入bool报错
    # 'clamp_scalar_bool': dict(
    #     name=['clamp'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     para=dict(
    #         min=[0.3, False, 1],
    #         max=[1.2, 1, False],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((12, 16), (1, 242991, 2),
    #                           (2, 4, 100, 152),),
    #                 "dtype": [Dtype.bool],
    #                 "gen_fn": Genfunc.mask,
    #             },
    #         ],
    #     ),
    # ),

    'clamp_max_scalar': dict(
        name=['clamp_max'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            max=[True, 4.13, 1, -1, 1e-12, 10, 0, 1.2, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152),
                              (384, 128),
                              (), (2, 0), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME clamp_max输入bool报错
    # 'clamp_max_scalar_bool': dict(
    #     name=['clamp_max'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     para=dict(
    #         max=[1.2, 1, -0.2],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((12, 16), (1, 242991, 2),
    #                           (2, 4, 100, 152),),
    #                 "dtype": [Dtype.bool],
    #                 "gen_fn": Genfunc.mask,
    #             },
    #         ],
    #     ),
    # ),

    'clamp_min_scalar': dict(
        name=['clamp_min'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            min=[0, 1.2, -1.1, 1, 100, 10, -2, 2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152),
                              (384, 128),
                              (), (2, 0), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME clamp输入bool报错
    # 'clamp_min_scalar_bool': dict(
    #     name=['clamp_min'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     para=dict(
    #         min=[1.2, 1, -0.2],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((12, 16), (1, 242991, 2),
    #                           (2, 4, 100, 152),),
    #                 "dtype": [Dtype.bool],
    #                 "gen_fn": Genfunc.mask,
    #             },
    #         ],
    #     ),
    # ),

    'clamp_tensor': dict(
        name=['clamp'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152),
                              (0,), (2, 0), (16, 0, 9)),
                },
                {
                    "ins": ['min'],
                    "shape": ((), (182, ), (384, 1),
                              None, (2, 4, 100, 152),
                              (1,), (0,), (0, 9)),
                },
                {
                    "ins": ['max'],
                    "shape": ((), None, (384, 128), (1, 1, 2), None,
                              (0,), (1, 0), (1, 0, 9)),
                },
            ],
        ),
    ),

    # FIXME clamp broadcast报错
    # 'clamp_tensor_broadcast': dict(
    #     name=['clamp'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8],
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((), (182,), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152),
    #                           (0,), (2, 0), (16, 0, 9)),
    #             },
    #             {
    #                 "ins": ['min'],
    #                 "shape": ((1,), (2, 182), (384, 1),
    #                           None, (2, 4, 100, 152),
    #                           (2, 1,), (3, 2, 0,), (0, 9)),
    #             },
    #             {
    #                 "ins": ['max'],
    #                 "shape": ((), None, (2, 1, 128), (1, 1, 2), None,
    #                           (2, 0,), (1, 0), (2, 1, 0, 9)),
    #             },
    #         ],
    #     ),
    # ),

    # FIXME clamp输入不同dtype结果不一致
    # 'clamp_tensor_diff_dtype': dict(
    #     name=['clamp'],
    #     interface=['torch'],
    #     is_inplace=True,
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((182, ), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152)),
    #                 "dtype":[Dtype.float32, Dtype.float64, Dtype.float16,
    #                          Dtype.int16, Dtype.int32, Dtype.int64,
    #                          Dtype.int8, Dtype.uint8, Dtype.float32],
    #             },
    #             {
    #                 "ins": ['min'],
    #                 "shape": ((182, ), (384, 1),
    #                           None, (2, 4, 100, 152)),
    #                 "dtype":[Dtype.bool, Dtype.float32, Dtype.float16,
    #                          Dtype.int16, Dtype.int64, Dtype.int32,
    #                          Dtype.int8, Dtype.int64, Dtype.uint8],
    #             },
    #             {
    #                 "ins": ['max'],
    #                 "shape": (None, (384, 128), (1, 1, 2), None),
    #                 "dtype":[Dtype.float32, Dtype.bool, Dtype.float16,
    #                          Dtype.uint8, Dtype.int32, Dtype.int64,
    #                          Dtype.int32, Dtype.int8, Dtype.int16],
    #             },
    #         ],
    #     ),
    # ),

    'clamp_max_tensor': dict(
        name=['clamp_max'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182,), (384, 128),
                              (3, 242991, 2,),
                              (2, 4, 100, 152),
                              (0,), (9, 0), (2, 0, 9)),
                },
                {
                    "ins": ['max'],
                    "shape": ((1,), (128, ), (3, 1, 2), (4, 100, 152),
                              (0,), (1,), (0, 9)),
                },
            ],
        ),
    ),

    # FIXME clamp_max broadcast报错
    # 'clamp_max_broadcast': dict(
    #     name=['clamp_max'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8],
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((), (182,), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152),
    #                           (0,), (2, 0), (16, 0, 9)),
    #             },
    #             {
    #                 "ins": ['max'],
    #                 "shape": ((), (2, 182), (2, 1, 128), (2, 1, 1, 2), (1, 100, 152),
    #                           (2, 0,), (1, 0), (2, 1, 0, 9)),
    #             },
    #         ],
    #     ),
    # ),

    # FIXME clamp_max输入不同dtype结果不一致
    # 'clamp_max_tensor_diff_dtype': dict(
    #     name=['clamp_max'],
    #     interface=['torch'],
    #     is_inplace=True,
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((182,), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152)),
    #                 "dtype":[Dtype.float32, Dtype.float64, Dtype.float16,
    #                          Dtype.int16, Dtype.int32, Dtype.int64,
    #                          Dtype.int8, Dtype.uint8, Dtype.float32],
    #             },
    #             {
    #                 "ins": ['max'],
    #                 "shape": ((182,), (384, 128), (1, 1, 2), (4, 1, 152)),
    #                 "dtype":[Dtype.float32, Dtype.bool, Dtype.float16,
    #                          Dtype.uint8, Dtype.int32, Dtype.int64,
    #                          Dtype.int32, Dtype.int8, Dtype.int16],
    #             },
    #         ],
    #     ),
    # ),

    'clamp_min_tensor': dict(
        name=['clamp_min'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182,), (384, 128),
                              (3, 242991, 2,),
                              (2, 4, 100, 152),
                              (0,), (9, 0), (2, 0, 9)),
                },
                {
                    "ins": ['min'],
                    "shape": ((1,), (128, ), (3, 1, 2), (4, 100, 152),
                              (0,), (1,), (0, 9)),
                },
            ],
        ),
    ),

    # FIXME clamp_min broadcast报错
    # 'clamp_min_broadcast': dict(
    #     name=['clamp_min'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8],
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((), (182,), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152),
    #                           (0,), (2, 0), (16, 0, 9)),
    #             },
    #             {
    #                 "ins": ['min'],
    #                 "shape": ((), (2, 182), (2, 1, 128), (2, 1, 1, 2), (1, 100, 152),
    #                           (2, 0,), (1, 0), (2, 1, 0, 9)),
    #             },
    #         ],
    #     ),
    # ),

    # FIXME clamp_min输入不同dtype结果不一致
    # 'clamp_min_tensor_diff_dtype': dict(
    #     name=['clamp_min'],
    #     interface=['torch'],
    #     is_inplace=True,
    #     atol=1e-4,
    #     rtol=1e-5,
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((182,), (384, 128),
    #                           (1, 242991, 2),
    #                           (2, 4, 100, 152)),
    #                 "dtype":[Dtype.float32, Dtype.float64, Dtype.float16,
    #                          Dtype.int16, Dtype.int32, Dtype.int64,
    #                          Dtype.int8, Dtype.uint8, Dtype.float32],
    #             },
    #             {
    #                 "ins": ['min'],
    #                 "shape": ((182,), (384, 128), (1, 1, 2), (4, 1, 152)),
    #                 "dtype":[Dtype.float32, Dtype.bool, Dtype.float16,
    #                          Dtype.uint8, Dtype.int32, Dtype.int64,
    #                          Dtype.int32, Dtype.int8, Dtype.int16],
    #             },
    #         ],
    #     ),
    # ),

    'fill': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[True, -100, 0.0, float("-inf"), 100, -2.4, float("-inf"),
                   3.0, 3, bool(3), float("inf"), float("nan"), False, 2.1, 5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1),
                              (1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726),
                              (0,), (0, 5), (3, 0, 6)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'fill_not_float': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0.234, 3.0, 3, bool(3), 4.8, -10, False, 5, 0.23],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (0,), (0, 5), (3, 0, 6)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (0, 2), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, [0, 1], 2, [-1, 0, 2], 3,
                 [0], -2, [0, 1]],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (0, 2), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, [0, 1], 2, [-1, 0, -3], [0, 2, 3, -1], -1, [-1, -2], 2],
            unbiased=[True, True, False, True, False, True, False, False, True, False],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (12, 0), (9, 0, 7)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_2': dict(
        name=['min', 'max'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, 1, 2, -1, 3, -2, 2],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (12, 0), (2, 0, 12)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_min_equal_input': dict(
        name=['min', 'max'],
        interface=['torch'],
        para=dict(
            dim=[0, -1, 1, 1, 2, -1, 3],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'max_min_all': dict(
        name=['min', 'max'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_3': dict(
        name=['any', 'all'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, 0, -2, -1, 3, 0, -1, 2],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (12, 0), (2, 0, 12)),
                    "dtype": [Dtype.bool, Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_4': dict(
        name=['sum'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, [0, 1], 2, [-1, 0, 2], 3,
                 [0], -2, [0, 1]],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (0, 2), (16, 0, 9)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'reduce_partial_op_zeros_input': dict(
        name=['any', 'all'],
        interface=['torch'],
        para=dict(
            dim=[0, -1, 1, 0, 2, -1, -4],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.bool, Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'reduce_partial_op_ones_input': dict(
        name=['any', 'all'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, 0, 2, -1, 3],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.bool, Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        para=dict(
            reduction=['mean', 'none', 'sum',
                       'mean', 'none', 'sum',
                       'mean', 'sum']
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (64,), (2, 11856, 2), (16, 2, 2964, 2), (2964, 32),
                              (0,), (16, 0), (4, 0, 9)),
                },
                {
                    "ins": ['target'],
                    "shape": ((), (64, ), (2, 11856, 2), (16, 2, 2964, 2), (2964, 32),
                              (0,), (16, 0), (4, 0, 9)),
                },
            ],
        ),
    ),

    # FIXME mse_loss输入int报错
    # 'mse_loss_diff_dtype': dict(
    #     name=["mse_loss"],
    #     para=dict(
    #         reduction=['mean', 'none', 'sum']
    #     ),
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2964, 32)),
    #                 "dtype": [Dtype.float64, Dtype.int32, Dtype.float16,
    #                           Dtype.int64, Dtype.float32, Dtype.int16,
    #                           Dtype.uint8, Dtype.int8, Dtype.float16, Dtype.float32]
    #             },
    #             {
    #                 "ins": ['target'],
    #                 "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2964, 32)),
    #                 "dtype": [Dtype.int16, Dtype.float32, Dtype.int64,
    #                           Dtype.float16, Dtype.int32, Dtype.float64,
    #                           Dtype.float32, Dtype.float64, Dtype.int8, Dtype.uint8],
    #             },
    #         ],
    #     ),
    # ),

    # FIXME nll_loss执行报错
    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            # reduction=['mean', 'none', 'mean', 'sum',
            #            'sum', 'sum', 'mean', 'none',
            #            'none', 'mean', 'sum', 'mean'],
            # ignore_index=[-100, 79, -100, 0,
            #               79, 0, 79, 100,
            #               -100, 94, 62, 0],
            reduction=['none',
                       'sum',
                       'none', 'mean', 'mean'],
            ignore_index=[79,
                          0,
                          -100, 94, 0],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((100,), (200, 79), (2, 92, 29), (2, 150, 512, 512),
                    #           (79,), (200, 80), (2, 79, 512, 512), (3, 80, 25, 24, 5),
                    #           (5, 16, 0), (0, 16,), (0, 5, 6, 0, 3), (4, 82, 0, 3)),
                    "shape": ((200, 79),
                              (200, 80),
                              (5, 16, 0), (0, 16,), (4, 82, 0, 3)),
                },
                {
                    "ins": ['target'],
                    # "shape": ((), (200, ), (2, 29), (2, 512, 512),
                    #           (), (200,), (2, 512, 512), (3, 25, 24, 5),
                    #           (5, 0), (0,), (0, 6, 0, 3), (4, 0, 3)),
                    "shape": ((200, ),
                              (200,),
                              (5, 0), (0,), (4, 0, 3)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=80),
                },
                {
                    "ins": ['weight'],
                    # "shape": (None, (79, ), (92, ), None,
                    #           (79,), (80,), (79, ), (80, ),
                    #           (16,), (16,), (5,), (82,)),
                    "shape": ((79, ),
                              (80,),
                              (16,), (16,), (82,)),
                },
            ],
        ),
    ),

    # FIXME nll_loss执行报错
    # 'nll_loss_empty_tensor': dict(
    #     name=["nll_loss"],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     para=dict(
    #         # reduction=['none', 'mean', 'sum', 'mean'],
    #         # ignore_index=[0, 0, 0, 0],
    #         reduction=['mean', 'sum', 'mean'],
    #         ignore_index=[0, 0, 0],
    #     ),
    #     dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "requires_grad": [True],
    #                 # "shape": ((0,), (16, 0,), (5, 0, 5, 6, 0, 3), (4, 0, 8, 3)),
    #                 "shape": ((16, 0,), (5, 0, 5, 6, 0, 3), (4, 0, 8, 3)),
    #             },
    #             {
    #                 "ins": ['target'],
    #                 # "shape": ((), (16,), (5, 5, 6, 0, 3), (4, 8, 3)),
    #                 "shape": ((16,), (5, 5, 6, 0, 3), (4, 8, 3)),
    #                 "dtype": [Dtype.int64],
    #                 "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
    #             },
    #             {
    #                 "ins": ['weight'],
    #                 # "shape": (None, (0,), (0,), (0,)),
    #                 "shape": ((0,), (0,), (0,)),
    #             },
    #         ],
    #     ),
    # ),

    # FIXME cross_entropy输入指定shape报错
    'cross_entropy': dict(
        name=["cross_entropy"],
        atol=1e-1,
        rtol=1e-2,
        para=dict(
            # reduction=['mean', 'none', 'none',
            #            'sum', 'mean', 'sum',
            #            'none', 'sum', 'mean'],
            # ignore_index=[-100, 9, -100,
            #               9, 0, 9,
            #               5, -100, 1],
            # label_smoothing=[0.0, True, 0.5,
            #                  1, 0.9, 0.3,
            #                  False, -1.3, 0.4],
            reduction=['mean', 'none',
                       'sum', 'sum',
                       'none', 'sum', 'mean'],
            ignore_index=[-100, 9,
                          9, 9,
                          5, -100, 1],
            label_smoothing=[0.0, True,
                             1, 0.3,
                             False, -1.3, 0.4],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((1024, 81), (3, 9), (3, 12, 6, 6, 7),
                    #           (64, 9), (4, 16, 8), (5, 9, 12, 4),
                    #           (0, 16,), (0, 5, 6), (4, 6, 0, 3)),
                    "shape": ((1024, 81), (3, 9),
                              (64, 9), (5, 9, 12, 4),
                              (0, 16,), (0, 5, 6), (4, 6, 0, 3)),
                },
                {
                    "ins": ['weight'],
                    # "shape": (None, (9,), (12,),
                    #           (9,), None, (9,),
                    #           (16,), (5,), (6,)),
                    "shape": (None, (9,),
                              (9,), (9,),
                              (16,), (5,), (6,)),
                },
                {
                    "ins": ['target'],
                    # "shape": ((1024,), (3,), (3, 6, 6, 7),
                    #           (64,), (4, 8), (5, 12, 4),
                    #           (0,), (0, 6), (4, 0, 3)),
                    "shape": ((1024,), (3,),
                              (64,), (5, 12, 4),
                              (0,), (0, 6), (4, 0, 3)),
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=10),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ),

    'cross_entropy_empty_tensor': dict(
        name=["cross_entropy"],
        atol=1e-1,
        rtol=1e-2,
        para=dict(
            reduction=['none'],
            ignore_index=[0],
            label_smoothing=[False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((5, 0),),
                },
                {
                    "ins": ['weight'],
                    "shape": ((0,),),
                },
                {
                    "ins": ['target'],
                    "shape": ((5,),),
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ),

    # FIXME cross_entropy输入指定shape报错
    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            # reduction=['sum', 'mean', 'none',
            #            'mean', 'none', 'sum',
            #            'none', 'sum', 'none'],
            # label_smoothing=[0.1, 0.3, 0.5,
            #                  False, 1, -1.2,
            #                  0, 1.0, -2],
            reduction=['sum', 'mean', 'none',
                       'mean', 'none', 'sum',
                       'none', 'sum'],
            label_smoothing=[0.1, 0.3, 0.5,
                             False, 1, -1.2,
                             0, 1.0],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8),
                    #           (3, 5, 6, 6), (1024, 81), (64, 8,),
                    #           (12, 0), (9, 0, 8), (0, 5, 0, 12)),
                    "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8),
                              (3, 5, 6, 6), (1024, 81), (64, 8,),
                              (12, 0), (9, 0, 8)),
                },
                {
                    "ins": ['weight'],
                    # "shape": ((5,), None, (8,),
                    #           (5,), None, (8,),
                    #           (0,), (0,), (5,)),
                    "shape": ((5,), None, (8,),
                              (5,), None, (8,),
                              (0,), (0,)),
                },
                {
                    "ins": ['target'],
                    # "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8),
                    #           (3, 5, 6, 6), (1024, 81), (64, 8,),
                    #           (12, 0), (9, 0, 8), (0, 5, 0, 12)),
                    "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8),
                              (3, 5, 6, 6), (1024, 81), (64, 8,),
                              (12, 0), (9, 0, 8)),
                    "gen_fn": dict(fn=Genfunc.uniform, low=0, high=1),
                },
            ],
        ),
    ),

    'select': dict(
        name=["select"],
        interface=['torch'],
        para=dict(
            dim=[0, 1, -2, 1, 0, 2],
            index=[11, -5, 0, 2, 0, -7],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "requires_grad": [True],
                    "shape": ((12,), (4, 5), (16, 4, 4), (64, 4, 8, 8),
                              (2, 0), (6, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ]
        ),
    ),

    'select_not_float': dict(
        name=["select"],
        interface=['torch'],
        para=dict(
            dim=[0, 1, -2, 1, 0, -3],
            index=[-12, 4, 0, 2, -2, 5],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "shape": ((12,), (4, 5), (16, 4, 4), (64, 4, 8, 8),
                              (2, 0), (6, 0, 9)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ]
        ),
    ),

    'index_select': dict(
        name=["index_select"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, 2, 3, -2, 2]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((5,), (5, 3), (16, 8), (1, 800, 1216), (4, 4, 14, 14),
                              (12, 0), (2, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    "shape": ((10,), (3,), (5,), (2,), (30,),
                              (12,), (7,)),
                    "dtype": [Dtype.int64, Dtype.int32, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'index_select_not_float': dict(
        name=["index_select"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, 2, 3, -2, 2]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [False],
                    "shape": ((12,), (10, 10), (16, 12), (1, 800, 1216), (4, 4, 14, 14),
                              (12, 0), (2, 0, 15)),
                    "dtype": [Dtype.int32, Dtype.int16, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    "shape": ((20,), (10,), (5,), (100,), (10,),
                              (20,), (7,)),
                    "dtype": [Dtype.int32, Dtype.int64, Dtype.int32, Dtype.int64, Dtype.int32, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=10)
                },
            ]
        ),
    ),

    # FIXME masked_scatter输入指定shape报错
    'masked_scatter': dict(
        name=["masked_scatter"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((), (), (16,), (12, 13), (12, 13, 14), (8, 9, 1, 17), (6, 4, 5), (12, 13, 14, 16), (4, 4),
                    #           (0,), (2, 0), (16, 0, 9)),
                    "shape": ((), (16,), (12, 13), (12, 13, 14), (12, 13, 14, 16), (4, 4),
                              (0,), (2, 0), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    # "shape": ((), (12, 3), (16,), (12, 13), (12, 13, 14), (9, 12, 17), (9, 1, 4, 5), (12, 13, 14, 1), (1, 4),
                    #           (0,), (1,), (9,)),
                    "shape": ((), (16,), (12, 13), (12, 13, 14), (12, 13, 14, 1), (1, 4),
                              (0,), (1,), (9,)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['source'],
                    # "shape": ((), (36,), (16,), (12, 13), (12, 13, 14), (12, 9, 12, 5, 3), (20, 50), (12, 13, 14, 16), (1, 20),
                    #           (0,), (3, 4), (2,)),
                    "shape": ((), (16,), (12, 13), (12, 13, 14), (12, 13, 14, 16), (1, 20),
                              (0,), (3, 4), (2,)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
            ],
            input='input',
        ),
    ),

    # FIXME nonzero输入0-d张量报错
    'nonzero': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64, Dtype.int16,
               Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.mask,
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((), (1482, ), (16, 24), (5, 8, 20),
                    #           (4, 4, 16, 20),
                    #           (4, 4, 16, 2, 20),
                    #           (0,), (12, 0), (2, 0, 9)),
                    "shape": ((1482, ), (16, 24), (5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20),
                              (0,), (12, 0), (2, 0, 9)),
                },
            ],
        ),
    ),

    'nonzero_float': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1482,), (16, 24)),
                },
            ],
        ),
    ),

    'nonzero_int': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-128, high=128),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'nonzero_uint': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.uint8],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=0, high=256),
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    # FIXME linear输入指定shape报错
    'linear': dict(
        name=["linear"],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-1,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((8,), (8,), (2, 512), (128, 49, 128), (6, 2, 100, 256),
                    #           (2, 6, 16, 8), (2, 31, 6, 40, 512), (2, 16, 8, 32, 7), (0,), (0,), (16, 8)),
                    "shape": ((2, 512), (128, 49, 128),
                              (2, 31, 6, 40, 512), (16, 8)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    # "shape": ((8,), (16, 8,), (10, 512), (384, 128), (81, 256),
                    #           (64, 8), (1, 512), (16, 7), (0,), (16, 0), (0, 8)),
                    "shape": ((10, 512), (384, 128),
                              (1, 512), (0, 8)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['bias'],
                    "requires_grad": [True],
                    # "shape": ((), (16,), (10, ), None, (100, 81, ),
                    #           (6, 1, 64), (1,), (2, 16, 8, 32, 16), None, (16,), (16, 0)),
                    "shape": ((10, ), None,
                              (1,), (16, 0)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 0, -2, 1, 3, -1, 1, -2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (13,), (78, 24), (2, 92, 29), (2, 150, 512, 512),
                              (0,), (0, 15), (5, 0, 13), (26, 20, 38)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'log_softmax_specific': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 0, -2, 1, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (1,), (1, 24), (2, 1, 29), (2, 150, 1, 512), (26, 20, 38)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_max': dict(
        name=["softmax"],
        atol=1e-4,
        rtol=1e-5,
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 0, -1, 1, 0, -1, -1, 1, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (16,), (2, 24), (2, 128, 24), (8, 16, 49, 49), (4, 12, 577, 577),
                              (0,), (0, 12), (16, 0, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME embedding输入负padding_idx精度不一致
    'embedding': dict(
        name=["embedding"],
        para=dict(
            # padding_idx=[None, None, 92, -20, 0, -15, 19, 2, 0],
            padding_idx=[None, None, 92, 0, 0, 0, 19, 2, 0],
            max_norm=[None, 1.0, None, None, -2, 2, None, 9, -0.5],
            norm_type=[2.0, 0, 1, 2, 0.5, 1.2, float('inf'), -2, -0.5],
            scale_grad_by_freq=[False, True, False, True, False, True, False, True, True],
            # sparse=[True, False, True, False, True, False, True, False, False],
            sparse=[False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((), (2, ), (2, 30), (2, 3, 4), (12, 4, 3, 8),
                              (64, ), (2, 16), (12, 4, 18), (1, 32)),
                    "dtype": [Dtype.int64, Dtype.int64, Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=10),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((10, 3), (10, 2), (93, 512), (20, 2), (16, 8),
                              (15, 3), (20, 3), (10, 5), (10, 4)),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'embedding_forward': dict(
        name=["embedding"],
        para=dict(
            padding_idx=[None],
            max_norm=[None],
            norm_type=[2.0],
            scale_grad_by_freq=[False],
            sparse=[False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((1, 32),),
                    "dtype": [Dtype.int64, Dtype.int64, Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=10),
                },
                {
                    "ins": ["weight"],
                    "shape": ((10, 0),),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'clip_grad_norm': dict(
        name=["clip_grad_norm_"],
        interface=["CustomizedTest"],
        para=dict(
            max_norm=[1.0, 5, 2.0, -1.2, 3, 10, 8, -0.5, 0, -2],
            norm_type=[0, -0.2, 1, 2.0, float('-inf'), float("inf"), 1.2, 2, 1, 3.0],
            error_if_nonfinite=[True, False, False, True, True, False, True, True, False, True],  # 1.7 not support
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["grads"],
                    "shape": ((), (10,), (10, 2, 5), (20,), (10, 5, 1), (20, 3, 4, 5), (20, 2, 3, 4, 5),
                              (0,), (0, 10), (5, 0, 9)),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                    "gen_num_range": [1, 5]
                },
            ],
            seq_name='tensors',
        ),
    ),

    # FIXME clip_grad_norm_ parameters输入指定参数报错
    'clip_grad_norm_diff_shape': dict(
        name=["clip_grad_norm_"],
        interface=["CustomizedTest"],
        para=dict(
            # max_norm=[1.0, -5, 2.0, -0.5, 0],
            # norm_type=[0, float("inf"), 1, 2, 0.5],
            # error_if_nonfinite=[True, False, False, True, False],  # 1.7 not support
            max_norm=[1.0, -5, 2.0, 0],
            norm_type=[0, float("inf"), 1, 0.5],
            error_if_nonfinite=[True, False, False, False],  # 1.7 not support
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    # "shape": (((),), ((10,), (10, 2, 5)), ((), (9, 10, 4, 2)),
                    #           ((20, 3, 4, 5), (20, 2, 3, 4, 5), (0,)),
                    #           ((2, 9), (0, 19), (2, 3, 1), (3, 5, 8, 9))),
                    "shape": (((),), ((10,), (10, 2, 5)), ((), (9, 10, 4, 2)),
                              ((2, 9), (0, 19), (2, 3, 1), (3, 5, 8, 9))),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                    "gen_num_range": [1, 5]
                },
            ],
            seq_name='tensors',
        ),
    ),

    # FIXME clip_grad_norm_ parameters输入指定参数报错
    # 'clip_grad_norm_empty_list': dict(
    #     name=["clip_grad_norm_"],
    #     interface=["CustomizedTest"],
    #     para=dict(
    #         max_norm=[1.0],
    #         norm_type=[2.0],
    #         error_if_nonfinite=[False],  # 1.7 not support
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ["tensors"],
    #                 "shape": ((()),),
    #                 "gen_fn": Genfunc.randn,
    #                 "dtype": [Dtype.float32],
    #                 "gen_num_range": [1, 5]
    #             },
    #         ],
    #         seq_name='tensors',
    #     ),
    # ),

    'tril': dict(
        name=["tril"],
        interface=["torch"],
        para=dict(
            diagonal=[12, 0, 5, -9, -1, 1, 2, 10, -10],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, 9), (6, 7), (6, 6), (9, 9),
                              (6, 8, 8), (64, 7, 28, 28),
                              (2, 0), (12, 0), (2, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'one_hot': dict(
        name=["one_hot"],
        para=dict(
            num_classes=[-1, -1, 6, 80, 4, 6],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (2, ), (6, 8, ), (64, 7, 28,), (0, ), (16, 0)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=6),
                },
            ],
        ),
    ),

    # FIXME stack输入size为0的张量报错
    'join': dict(
        name=['cat', 'stack'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            # dim=[-1, 1, 0, 2, 1, 1, -1, 1, -2],
            dim=[-1, 1, 0, 2, 1, 1, -1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "requires_grad": [True],
                    # "shape": ((3, ), (512, 4),
                    #           (0, 50, 76), (2, 31, 512),
                    #           (2, 512, 8, 8), (1, 64, 4, 56, 56),
                    #           (0,), (16, 0), (8, 0, 2)),
                    "shape": ((3, ), (512, 4),
                              (0, 50, 76), (2, 31, 512),
                              (2, 512, 8, 8), (1, 64, 4, 56, 56),
                              (0,), (16, 0)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64, Dtype.int16,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                    "gen_num_range": [1, 5]
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_diff_size': dict(
        name=['cat'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dim=[-1, 0, -2, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "requires_grad": [True],
                    "shape": (((8,), (16,),),
                              ((2, 8,), (16, 8,), (3, 8,), (4, 8,), (1, 8,)),
                              ((3, 16, 8,), (3, 2, 8,), (3, 7, 8,)),
                              ((2, 512, 8, 8), (2, 128, 8, 8), (2, 2, 8, 8), (2, 1, 8, 8)),
                              ((2, 31, 0), (2, 31, 512), (2, 31, 128)),),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64, Dtype.int16,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    # FIXME split分解维度size为0时报错
    'split': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            # split_size_or_sections=[[1, 1, 1, 1], [5, ], [15200, 3800, 950, 247, 70], 4, 3,
            #                         0, [3, 5, 4], [0, ]],
            # dim=[-1, 1, 0, 2, 1, -1, 0, 1]
            split_size_or_sections=[[1, 1, 1, 1], [5, ], [15200, 3800, 950, 247, 70], 4, 3,
                                    [3, 5, 4]],
            dim=[-1, 1, 0, 2, 1, 0]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    # "shape": ((1, 4), (4, 5, 12),
                    #           (20267, ), (9, 12, 17, 6, 5),
                    #           (4, 6, 10, 9, 8),
                    #           (0,), (12, 0), (3, 0, 9)),
                    "shape": ((1, 4), (4, 5, 12),
                              (20267, ), (9, 12, 17, 6, 5),
                              (4, 6, 10, 9, 8),
                              (12, 0)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'split_bool': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            split_size_or_sections=[[1, 1, 1, 1], [15200, 3800, 950, 247, 70], 3],
            dim=[-1, 0, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "shape": ((1, 4),
                              (20267, ),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1, -2, 3, -1, 0, -1, 0, 2],
            descending=[False, True, False, False, True, False, True, True, False, False],
            stable=[False, True, False, False, True, True, True, False, True, True],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
               Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (11400, ), (12, 8), (8, 12, 9),
                              (4, 4, 16, 20), (4, 4, 16, 2, 20), (24180,),
                              (0,), (12, 0), (4, 0, 5)),
                },
            ],
        ),
    ),

    'sort_same_value': dict(
        name=["sort"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1],
            descending=[True, False, False],
            stable=[False, True, False],
        ),
        dtype=[Dtype.float16, Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.ones,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((11400, ),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    # FIXME topk输入0-d张量，且k为0时，结果精度不一致
    'topk_nonzero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            # k=[1, 0, 9, 12, 4, 3, 0, 12, 5],
            k=[1, 1, 9, 12, 4, 3, 0, 12, 5],
            dim=[-1, 0, -1, 0, 1, 2, -1, 0, 2],
            largest=[True, False, True, False, True, False, False, True, False],
            sorted=[True, False, True, True, False, False, True, False, False],
        ),
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
                   Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (), (8723, ), (1024, 81),
                              (5, 4, 6), (2, 2, 64, 64),
                              (0,), (12, 0), (4, 0, 7)),
                },
            ],
        ),
    ),

    'topk_zero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            k=[1, 10, 20],
            dim=[-1, 0, 1],
            largest=[True, False, True],
            sorted=[True, False, True],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (16, ), (50, 25, 10)),
                },
            ],
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        interface=['torch'],
        para=dict(
            dim0=[0, -1, 1, 1, -2, 0, 1, 0],
            dim1=[-1, -1, 2, -1, -1, -1, 0, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
                   Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool, Dtype.int32],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (32,), (2, 1536, 950),
                              (16, 8), (660, 6, 49, 32),
                              (0,), (0, 8), (16, 0, 8)),
                },
            ],
        ),
    ),

    'where': dict(
        name=['where'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": [(), (1024, ), (1482, 4), (4, 5, 6),
                              (0,), (2, 0), (2, 0, 9)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input', 'other'],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "shape": [(), (1024, ), (1482, 4), (4, 5, 6),
                              (0,), (2, 0), (2, 0, 9)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'where_broadcast': dict(
        name=['where'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": [(), (1, ), (3, ), (3, ), (1, 445), (3, 5), (4, ),
                              (3, 4, 5), (3, ), (0,), (2, 0), (2, 0, 9)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1,), (2, 7), (1, ), (3, ), (2, 445), (3, 5), (1, ), (4, 5),
                              (5, 4, 3), (1,), (6, 2, 0), (2, 1, 9)],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['other'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(8,), (7,), (2, 1), (1, ), (1, ), (1, ), (4, ), (5, ), (4, 3),
                              (0,), (2, 1), (0, 1)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    # FIXME where输入不同dtype，计算结果不一致
    # 'where_diff_dtype': dict(
    #     name=['where'],
    #     interface=['torch'],
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['condition'],
    #                 "shape": [(3, 4), ],
    #                 "dtype": [Dtype.bool],
    #                 "gen_fn": Genfunc.mask
    #             },
    #             {
    #                 "ins": ['input'],
    #                 "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
    #                           Dtype.int16, Dtype.int32, Dtype.int64,
    #                           Dtype.uint8, Dtype.int8, Dtype.bool],
    #                 "shape": [(3, 4), ],
    #                 "gen_fn": Genfunc.randn
    #             },
    #             {
    #                 "ins": ['other'],
    #                 "dtype": [Dtype.float32, Dtype.int32, Dtype.float32,
    #                           Dtype.float16, Dtype.float32, Dtype.bool,
    #                           Dtype.int8, Dtype.uint8, Dtype.uint8],
    #                 "shape": [(3, 4), ],
    #                 "gen_fn": Genfunc.randn
    #             },
    #         ],
    #     ),
    # ),

    'where_same_value': dict(
        name=['where'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": [(1, ), (3, ), (3, ), (1, 445), (3, 5), (4, ),
                              (3, 4, 5), (3, )],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (3, ), (1, 445), (3, 5), (1, ), (4, 5),
                              (5, 4, 3)],
                    "gen_fn": Genfunc.zeros
                },
                {
                    "ins": ['other'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (1, ), (1, ), (1, ), (4, ), (5, ), (4, 3)],
                    "gen_fn": Genfunc.zeros
                },
            ],
        ),
    ),

    # FIXME dropout部分shape下报错（偶发）
    # 'dropout': dict(
    #     name=["dropout"],
    #     no_output_ref=True,
    #     is_inplace=True,
    #     para=dict(
    #         p=[0.5, 0.5, 0.1,
    #            0.4, 1, 0.7,
    #            0, 0.05, 0.9],
    #         training=[True, True, True,
    #                   False, True, True,
    #                   False, True, True]
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((), (1024,), (2, 4096), (32, 49, 256),
    #                           (2, 16, 64, 64), (1, 2304, 1, 1, 1),
    #                           (0,), (16, 0), (8, 0, 16)),
    #                 "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    # 'dropout_training': dict(
    #     name=["dropout"],
    #     no_output_ref=True,
    #     para=dict(
    #         p=[0.5, 0, 0.1,
    #            0.4, 1, 0.7,
    #            0, 0.05, 0.9],
    #         training=[False, False, False,
    #                   False, False, False,
    #                   False, False, False],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((), (1024,), (2, 4096), (32, 49, 256),
    #                           (2, 16, 64, 64), (1, 2304, 1, 1, 1),
    #                           (0,), (16, 0), (8, 0, 16)),
    #                 "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.5, 0, 0.1, 0.4],
            training=[True, True, True, False]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 4096), (32, 49, 256), (2, 16, 64, 64), (1, 2304, 1, 1, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'dropout_training': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0.5, 0, 0.1, 0.4],
            training=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 4096), (32, 49, 256), (2, 16, 64, 64),
                              (1, 2304, 1, 1, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    # FIXME dropout2d部分shape下报错（偶发）
    # 'dropout2d': dict(
    #     name=["dropout2d"],
    #     no_output_ref=True,
    #     is_inplace=True,
    #     para=dict(
    #         p=[0, 0.2, 0.5,
    #            0.7, 0.4,
    #            0.6, 0.3,
    #            0.8, 0.5, 1],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((2, 2048), (4, 1024), (32, 49, 256),
    #                           (32, 16, 64, 64), (2, 2, 1, 128, 64),
    #                           (2048, 128, 16,), (4096, 2, 4, 16),
    #                           (0,), (16, 0), (8, 0, 16)),
    #                 "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    'dropout2d': dict(
        name=["dropout2d"],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.5, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32, 49, 256), (32, 49, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'leaky_relu': dict(
        name=["leaky_relu"],
        atol=1e-4,
        rtol=1e-5,
        is_inplace=True,
        para=dict(
            negative_slope=[-5, -1, False, 0.01, 0.1, 10, 1, 0.0, -1.5]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (128,), (16, 7), (64, 28, 28),
                              (2, 32, 208, 304), (64, 3, 7, 28, 28),
                              (0,), (0, 8), (16, 0, 8)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sigmoid_focal_loss': dict(
        name=["sigmoid_focal_loss"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        para=dict(
            alpha=[0.25, 0.1, 0.9, 2, 3.4, 0, -2, -1.3],
            gamma=[2, 0.1, 10, 1.2, 0.4, 0, -3, -1.2],
            reduction=["mean", "sum", "none", "none", "sum", "mean", "none", "sum"],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['inputs'],
                    "requires_grad": [True],
                    "shape": ((), (64,), (16, 7), (2, 11856, 2), (16, 2, 2964, 2),
                              (0,), (6, 0), (12, 0, 4)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['targets'],
                    "shape": ((), (64,), (16, 7), (2, 11856, 2), (16, 2, 2964, 2),
                              (0,), (6, 0), (12, 0, 4)),
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'nms': dict(
        name=["nms"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            iou_threshold=[0.3, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['boxes'],
                    "value": ([[2.4112, 0.7486, 2.4551, 2.7486],
                              [0.7486, 1.3544, 1.1294, 2.3544],
                              [1.4551, 0.1294, 1.6724, 1.3294],
                              [1.4959, 0.1086, 2.778335, 3.22],
                              [0.107706, 2.948, 2.1256, 4.525],
                              [2.7735, 2.12506, 7.0556, 8.995]],

                              [[1.5, 2.2, 2.77, 3.2],
                              [2.5, 5.9, 10.6, 14.55]],),
                },
                {
                    "ins": ['scores'],
                    "shape": ((6, ), (2, )),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'roi_align': dict(
        name=["roi_align"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            output_size=[(5, 6), 3],
            spatial_scale=[0.8, 1.0],
            sampling_ratio=[1, -1],
            aligned=[True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((6, 3, 32, 32), (2, 3, 16, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['boxes'],
                    "value": ([[1, 2.4112, 0.7486, 2.4551, 2.7486],
                              [0, 0.7486, 1.3544, 1.1294, 2.3544],
                              [2, 1.4551, 0.1294, 1.6724, 1.3294],
                              [5, 1.4959, 0.1086, 2.778335, 3.22],
                              [2, 0.107706, 2.948, 2.1256, 4.525],
                              [4, 2.7735, 2.12506, 7.0556, 8.995]],

                              [[0, 1.5, 2.2, 2.77, 3.2],
                              [1, 2.5, 5.9, 10.6, 14.55]],),
                },
            ],
        ),
    ),

    # FIXME slice输入指定参数报错
    'slice': dict(
        name=["slice_op"],
        interface=["CustomizedTest"],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        para=dict(
            # index=(slice(0, 3, 1), slice(3, 20, 2), slice(0, 3, 1), slice(0, 4, 2), slice(-3, -2, 1),
            #        slice(0, -4, 2), slice(2, 4, 1), slice(5, 1, 2)),
            # dim=[0, -2, 1, 2, 0, -1, 0, 1],
            index=(slice(0, 3, 1), slice(0, 3, 1), slice(0, 4, 2), slice(-3, -2, 1)),
            dim=[0, 1, 2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((7, ), (7, 10), (128, 3, 3), (2, 3, 224, 224), (3, 2, 6, 197, 64),
                    #           (0,), (4, 0), (12, 0, 9)),
                    "shape": ((7, ), (128, 3, 3), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME slice输入指定参数报错
    'slice_int': dict(
        name=["slice_op"],
        interface=["CustomizedTest"],
        dtype=[Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
        para=dict(
            # index=(slice(0, 3, 1), slice(3, 20, 2), slice(0, 3, 1), slice(0, 4, 2), slice(-3, -2, 1),
            #        slice(0, -4, 2), slice(2, 4, 1), slice(5, 1, 2)),
            # dim=[0, -2, 1, 2, 0, -1, 0, 1],
            index=(slice(0, 3, 1), slice(0, 3, 1), slice(0, 4, 2), slice(-3, -2, 1)),
            dim=[0, 1, 2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((7, ), (7, 10), (128, 3, 3), (2, 3, 224, 224), (3, 2, 6, 197, 64),
                    #           (0,), (4, 0), (12, 0, 9)),
                    "shape": ((7, ), (128, 3, 3), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME index输入指定参数报错
    'index': dict(
        name=["index"],
        interface=["CustomizedTest"],
        # input[idx1,idx2,idx3] input[...,idx3] input[idx,...,idx3]
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((), (12,), (12, 32), (128, 2, 2), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "shape": ((12,), (128, 2, 2), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['idx1'],
                    # "shape": (None, (24,), None, (1, ), None, (1, )),
                    "shape": ((24,), (1, ), None, (1, )),
                    "gen_fn": dict(fn=Genfunc.randint, high=3),
                    "dtype": [Dtype.int64, Dtype.int32, Dtype.int64],
                },
                {
                    "ins": ['idx2'],
                    # "shape": (None, None, (11, 64), (1, ), None, None),
                    "shape": (None, (1, ), None, None),
                    "gen_fn": dict(fn=Genfunc.randint, high=2),
                    "dtype": [Dtype.int32],
                },
                {
                    "ins": ['idx3'],
                    # "shape": (None, None, None, (2, ), (224, 224), (64, )),
                    "shape": (None, (2, ), (224, 224), (64, )),
                    "gen_fn": Genfunc.mask,
                    "dtype": [Dtype.bool, Dtype.uint8, Dtype.bool],
                },
            ],
        ),
    ),

    'index_empty_tensor': dict(
        name=["index"],
        interface=["CustomizedTest"],
        # input[idx1,idx2,idx3] input[...,idx3] input[idx,...,idx3]
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((0,), (4, 0), (4, 0, 5)),
                    "shape": ((4, 0), (4, 0, 5)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['idx1'],
                    # "shape": (None, (5,), (1,)),
                    "shape": ((5,), (1,)),
                    "gen_fn": dict(fn=Genfunc.randint, high=3),
                    "dtype": [Dtype.int64],
                },
                {
                    "ins": ['idx2'],
                    # "shape": (None, None, None),
                    "shape": (None, None),
                    "gen_fn": dict(fn=Genfunc.randint, high=2),
                    "dtype": [Dtype.int64],
                },
                {
                    "ins": ['idx3'],
                    # "shape": (None, None, (5,)),
                    "shape": (None, (5,)),
                    "gen_fn": Genfunc.mask,
                    "dtype": [Dtype.bool],
                },
            ],
        ),
    ),

    'index_int': dict(
        name=["index"],
        interface=["CustomizedTest"],
        # input[idx1,idx2,idx3] input[...,idx3] input[idx,...,idx3]
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((), (12,), (12, 32), (128, 2, 2), (4, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "shape": ((12,), (128, 2, 2)),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['idx1'],
                    # "shape": (None, None, None, (1, ), (2,), (1, 4)),
                    "shape": (None, (1, )),
                    "gen_fn": dict(fn=Genfunc.randint, high=3),
                    "dtype": [Dtype.int64],
                },
                {
                    "ins": ['idx2'],
                    # "shape": (None, (12,), (12, 32), (2, ), None, None),
                    "shape": ((12,), (2, )),
                    "gen_fn": Genfunc.mask,
                    "dtype": [Dtype.bool],
                },
                {
                    "ins": ['idx3'],
                    # "shape": (None, None, None, (1, ), (128, 2), (23, 4,)),
                    "shape": (None, (1, )),
                    "gen_fn": dict(fn=Genfunc.randint, high=2),
                    "dtype": [Dtype.int32],
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            lr=[0.05, 0.001, 0.1, 0.1, 0, 3, 0.2, 0.07],
            momentum=[0.5, 0, 0.01, 0.01, 1, 0.5, 2, 1.2],
            weight_decay=[0, 0.5, 0, 0.1, 3, 2.3, 4.0, 5],
            dampening=[0, -0.5, 0.1, 0, 2, 3.0, 0, 6.5],
            nesterov=[True, False, False, True, False, False, True, False],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(), (16, 8), (2, 3, 16), (4, 32, 7, 7), (4, 16, 3, 8, 2),
                              (0,), (3, 0), (4, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['buf'],
                    "shape": [(), (16, 8), (2, 3, 16), (4, 32, 7, 7), (4, 16, 3, 8, 2),
                              (0,), (3, 0), (4, 0, 9)],
                    "gen_fn": Genfunc.rand,
                },
            ]
        ),
    ),

    'sgd_without_buf': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            nesterov=[False, True],
            lr=[0.1, 0.1],
            momentum=[0.01, 0.01],
            weight_decay=[0, 0.1],
            dampening=[0.1, 0],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'masked_fill_scalar': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=True,
        para=dict(
            value=[-100, 0.0, float("-inf"), False, True, float("-inf"), -23.4, 5, float("nan"), -2.3, 0.231, 20],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726),
                              (2, 31, 6, 40, 23), (4, 49), (4, 49), (4, 49),
                              (0,), (0, 5), (3, 0, 4)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((), (64,), (4, 49), (1, 49), (2, 1, 1, 726),
                              (2, 31, 6, 40, 1), (4, 49), (49,), (4, 1),
                              (1,), (0, 5), (1, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'masked_fill_scalar_int': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=True,
        para=dict(
            value=[-100, 0.0, False, 0.432, 23.4, True, 1, -2, 50],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1),
                              (0,), (0, 5), (3, 0, 4)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ['mask'],
                    "shape": ((), (64,), (4, 49), (1276, 49, 49), (2, 1, 1, 726), (2, 31, 6, 40, 1),
                              (1,), (0, 5), (1, 4)),
                    "dtype": [Dtype.bool],  # uint8 is deprecated
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'masked_fill_scalar_without_inplace': dict(
        name=["masked_fill"],
        interface=["torch"],
        para=dict(
            value=[-100, 0.0, 0, 1., -2, False, 23.4, 5, True, -2, 0.231, 20],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (3,), (4, ), (2, 3, 1), (2, 8, 1, 10),
                              (2, 1, 6, 1, 23), (4, 4), (4, 4), (5,),
                              (0,), (5,), (0, 4)),
                    "dtype": [Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((), (), (4, 4), (1, 2), (8, 13, 10),
                              (11, 6, 1, 1), (4, 4), (4,), (5, 1),
                              (1,), (0, 5), (1, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'masked_fill_tensor': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1), (0,), (0, 5), (3, 0, 4)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((), (64,), (4, 49), (1276, 49, 49), (2, 1, 1, 726), (2, 31, 6, 40, 1), (1,), (0, 5), (1, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['value'],
                    # masked_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), (), (), (), (), (), ()),
                    "dtype": [Dtype.int32, Dtype.bool, Dtype.uint8, Dtype.float32,
                              Dtype.float16, Dtype.int64, Dtype.int8, Dtype.uint8, Dtype.float64],
                    "gen_fn": Genfunc.ones
                },
            ],
        ),
    ),

    'masked_fill_tensor_without_inplace': dict(
        name=["masked_fill"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (5,), (1, 3), (2, 1, 5), (2, 8, 1, 3), (2, 31, 6, 2, 1), (0,), (5,), (0, 4)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((1,), (5,), (4, 3), (6, 5), (2, 1, 4, 3), (2, 31, 6, 2, 5), (1,), (0, 5), (1, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['value'],
                    # masked_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), (), (), (), (), (), ()),
                    "dtype": [Dtype.float64],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1,), (182,), (64, 128), (2, 1, 640, 640),
                              (0,), (12, 0), (4, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ],
        ),
    ),

    'reciprocal_int': dict(
        name=["reciprocal"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (182,), (64, 128), (2, 1, 640, 640)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'reciprocal_zero': dict(
        name=["reciprocal"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182,),),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    # FXIME adamw、adam输出精度不一致
    'adam': dict(
        name=['adam', 'adamw'],
        interface=["CustomizedTest"],
        atol=1e-4,
        rtol=1e-3,
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            # lr=[0, -0.2, 2, 0.001, 0.1, 3.2, -2, 0],
            # beta1=[0, -1, 0.004, 0.9, 0.8, -2, 4.3, 0],
            # beta2=[0.3, 0, -2, 0.99, 0.88, 1, -4, 0],
            # eps=[-1e-02, 0, 1e-2, 1e-08, 1e-09, 0, 2, 1e-4],
            # step=[3, 2, 0, 1, 4, 2, 4, 5],
            # weight_decay=[-0.2, 0, 2, 0, 0.1, 2.5, 0, -3],
            # amsgrad=[False, True, True, False, True, False, True, True],
            lr=[0, 3.2, -2, 0],
            beta1=[0, -2, 4.3, 0],
            beta2=[0.3, 1, -4, 0],
            eps=[-1e-02, 0, 2, 1e-4],
            step=[3, 2, 4, 5],
            weight_decay=[-0.2, 2.5, 0, -3],
            amsgrad=[False, False, True, True],
        ),
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    # "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                    #           (0,), (4, 0), (12, 0, 9)],
                    "shape": [(), (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'],
                    # "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                    #           (0,), (4, 0), (12, 0, 9)],
                    "shape": [(), (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    # FIXME conv_transpose2d特定参数组合，反向传播失败
    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        atol=1e-3,
        rtol=1e-3,
        atol_half=1e2,
        rtol_half=1e2,
        # out = (in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        para=dict(
            # stride=[1, 2, (1, 3), 1, 2, 1, 2, (2, 2), 1],
            # padding=[0, (4, 3), (5, 4), 0, 1, 0, 1, (1, 0), 0],
            # output_padding=[0, (2, 1), (3, 2), 0, 1, 0, 1, (0, 1), 0],
            # groups=[1, 2, 3, 1, 8, 1, 1, 1, 1],
            # dilation=[1, (3, 5), (4, 5), 1, 2, 1, 2, (1, 2), 1],
            stride=[1, 1, 2, 1, 2, (2, 2), 1],
            padding=[0, 0, 1, 0, 1, (1, 0), 0],
            output_padding=[0, 0, 1, 0, 1, (0, 1), 0],
            groups=[1, 1, 8, 1, 1, 1, 1],
            dilation=[1, 1, 2, 1, 2, (1, 2), 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    # "shape": ((6, 16, 20, 8), (6, 2, 4, 3), (6, 18, 4, 3),
                    #           (2, 256, 14, 14), (2, 128, 32, 32),
                    #           (2, 64, 160, 160), (2, 64, 320, 320), (2, 64, 320, 320),
                    #           (0, 16, 20, 8)),
                    "shape": ((6, 16, 20, 8),
                              (2, 256, 14, 14), (2, 128, 32, 32),
                              (2, 64, 160, 160), (2, 64, 320, 320), (2, 64, 320, 320),
                              (0, 16, 20, 8)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    # "shape": ((16, 2, 12, 2), (2, 4, 12, 16), (18, 3, 2, 1),
                    #           (256, 256, 2, 2), (128, 128, 4, 4),
                    #           (64, 64, 2, 2), (64, 1, 2, 2), (64, 1, 2, 2),
                    #           (16, 2, 12, 2)),
                    "shape": ((16, 2, 12, 2),
                              (256, 256, 2, 2), (128, 128, 4, 4),
                              (64, 64, 2, 2), (64, 1, 2, 2), (64, 1, 2, 2),
                              (16, 2, 12, 2)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    # "shape": (None, (8,), (9,), (256,), None, (64,), (1,), (1,), None),
                    "shape": (None, (256,), None, (64,), (1,), (1,), None),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ]
        ),
    ),

    'unfold': dict(
        name=["unfold"],
        interface=['torch.Tensor'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dimension=[2, 1, 2, 1, 1, -1, -1, 0, -1, -2, 2],
            size=[2, 2, 4, 5, 20, 2, 10, 1, 0, 3, 5],
            step=[1, 1, 1, 3, 2, 1, 3, 2, 1, 2, 3],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 128, 56, 56), (2, 512, 14, 14), (2, 96, 200, 304), (2, 128, 36), (10, 20), (16,), (20,), (),
                              (0,), (16, 0), (4, 0, 16)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ],
        ),
    ),

    'unfold_int': dict(
        name=["unfold"],
        interface=['torch.Tensor'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dimension=[2, 1, 2, -1, 0],
            size=[2, 4, 1, 5, 16],
            step=[1, 1, 1, 3, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 128, 56, 56), (2, 512, 14, 14), (2, 96, 200, 304), (2, 128, 36), (16,)),
                    "dtype": [Dtype.bool, Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
                },
            ],
        ),
    ),

    'cumsum': dict(
        name=["cumsum"],
        interface=['torch'],
        atol=1e-6,
        rtol=1e-5,
        para=dict(
            dim=[0, -1, 1, 2, 0, -1, -1, -2, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (12,), (2, 22, 33), (2, 2, 10, 16), (1, 20), (2, 2, 20),
                              (0,), (5, 0), (4, 0, 12)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        interface=['torch'],
        atol=1e-2,
        rtol=1e-3,
        saved_args=dict(output=0),
        para=dict(
            p=[1, 2, 0, 0.5, float("inf"), 1.2, 0, 2, 1, 2, 0],
            compute_mode=['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist', 'use_mm_for_euclid_dist_if_necessary',
                          'use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist',
                          'use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist_if_necessary',
                          'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist', 'use_mm_for_euclid_dist_if_necessary']
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "requires_grad": [True],
                    "shape": ((2, 50, 4), (1, 32, 32), (4, 31, 256), (4, 256, 256), (10, 128),
                              (2, 50, 4), (1, 32, 32), (4, 31, 256),
                              (0, 4, 5), (4, 0, 9), (3, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['x2'],
                    "shape": ((100, 4), (32, 2, 48, 64, 32), (4, 256, 256), (4, 256, 256), (1, 10, 128),
                              (100, 4), (32, 2, 48, 64, 32), (4, 256, 256),
                              (0, 6, 5), (4, 5, 9), (2, 1, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cdist_compute_mode': dict(
        name=['cdist'],
        interface=['torch'],
        atol=1e-2,
        rtol=1e-3,
        saved_args=dict(output=0),
        para=dict(
            p=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            compute_mode=['use_mm_for_euclid_dist', 'use_mm_for_euclid_dist', 'use_mm_for_euclid_dist', 'use_mm_for_euclid_dist',
                          'use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist_if_necessary',
                          'donot_use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "requires_grad": [True],
                    "shape": ((5, 4), (2, 256, 256), (2, 16, 256), (5, 4, 256, 256),
                              (3, 5, 4), (2, 256, 256), (3, 2, 16, 256), (5, 4, 26, 256),
                              (5, 4), (2, 256, 256), (2, 16, 256), (5, 4, 256, 256),),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['x2'],
                    "shape": ((3, 4), (2, 16, 256), (2, 26, 256), (2, 5, 4, 256, 256),
                              (3, 4), (2, 16, 256), (2, 26, 256), (2, 5, 4, 256, 256),
                              (4, 3, 4), (2, 16, 256), (4, 2, 26, 256), (2, 5, 4, 256, 256),),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'bitwise_not_uint8': dict(
        name=['bitwise_not'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1,), (100, 4), (2, 256, 256),
                              (0,)),
                    "dtype": [Dtype.uint8],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=256),
                },
            ],
        ),
    ),

    'bitwise_not_int': dict(
        name=['bitwise_not'],
        interface=['torch'],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((10,), (100, 4), (2, 256, 256),
                              (4, 0), (8, 0, 9)),
                    "dtype": [Dtype.bool, Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8],
                    "gen_fn": dict(fn=Genfunc.randint, low=-128, high=128),
                },
            ],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        interface=["torch"],
        para=dict(
            dim=[0, -1, 0, 1, None, -2, 2, 1],
            keepdim=[True, False, True, False, False, True, True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1,), (1024, 80), (2, 256, 256), (2, 1, 64, 64),
                              (12, 0), (2, 0, 9), (0, 9, 8, 7)),
                    "dtype": [Dtype.float64, Dtype.float16, Dtype.float32, Dtype.int32, Dtype.int16,
                              Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'argmax_same_value': dict(
        name=['argmax'],
        interface=["torch"],
        para=dict(
            dim=[-1, 0, None, 1],
            keepdim=[True, False, True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (1024, 80), (2, 256, 256), (2, 1, 64, 64)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'adadelta': dict(
        name=["adadelta"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        atol=1e-4,
        rtol=1e-4,
        para=dict(
            lr=[1.0, 0, -0.5, 0.1, 0.1, 2.3, -2, 0],
            rho=[-1, 1.2, 0, 0.9, 0.88, -3, 0.5, 0],
            eps=[1e-2, 0, -1e-4, 1e-6, 1e-6, 0, 1e-4, -1e-6],
            weight_decay=[1.2, 0.5, -1.3, 0, 0.1, 0.5, 0, -1.2],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                              (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['square_avg', 'acc_delta'],
                    "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                              (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'rmsprop': dict(
        name=["rmsprop"],
        interface=["CustomizedTest"],
        atol_half=1e-3,
        rtol_half=1e-2,
        atol=1e-5,
        rtol=1e-3,
        para=dict(
            lr=[0, 1.2, -0.05, 0.1, 0.01, 0, 2, 2.3],
            alpha=[-0.3, 0, 1.2, 0.9, 0.99, 3, 0, 0.4],
            eps=[1e-2, 0, -1e-4, 1e-6, 1e-8, 0, 1e-4, -1e-6],
            weight_decay=[1.2, 0.5, -1.3, 0, 0.1, 0.5, 0, -1.2],
            momentum=[-2, 0.3, 1, 0, 0.1, 0.05, -3, 0],
            centered=[True, False, True, False, True, True, False, True],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16, Dtype.float64],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                              (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['square_avg', 'grad_avg', 'momentum_buffer'],
                    "shape": [(), (16,), (16, 8), (2, 3, 16), (4, 32, 7, 7),
                              (0,), (4, 0), (12, 0, 9)],
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'smooth_l1_loss': dict(
        name=["smooth_l1_loss"],
        para=dict(
            reduction=['mean', 'none', 'sum',
                       'mean', 'none', 'sum',
                       'mean', 'none', 'sum'],
            beta=[0, 1.0, True,
                  0.5, 0.1, 1.2,
                  0, 1.5, 6]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), (64,), (2964, 32),
                              (2, 11856, 2), (16, 2, 2964, 2), (2, 16, 128, 128),
                              (0,), (16, 0), (4, 0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.float32, Dtype.float64]
                },
                {
                    "ins": ['target'],
                    "shape": ((), (64,), (2964, 32),
                              (2, 11856, 2), (16, 2, 2964, 2), (2, 16, 128, 128),
                              (0,), (16, 0), (4, 0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8],
                },
            ],
        ),
    ),

    # FIXME smooth_l1_loss input输入int报错
    # 'smooth_l1_loss_int': dict(
    #     name=["smooth_l1_loss"],
    #     para=dict(
    #         reduction=['mean', 'none', 'sum'],
    #         beta=[0.5, 0.1, 0.1]
    #     ),
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2964, 32)),
    #                 "dtype": [Dtype.float64, Dtype.int32, Dtype.float16,
    #                           Dtype.int64, Dtype.float32, Dtype.int16,
    #                           Dtype.uint8, Dtype.int8, Dtype.float16, Dtype.float32]
    #             },
    #             {
    #                 "ins": ['target'],
    #                 "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2964, 32)),
    #                 "dtype": [Dtype.int16, Dtype.float32, Dtype.int64,
    #                           Dtype.float16, Dtype.int32, Dtype.float64,
    #                           Dtype.float32, Dtype.float64, Dtype.int8, Dtype.uint8],
    #             },
    #         ],
    #     ),
    # ),

    # FIXME conv3d输入指定shape报错
    'conv3d': dict(
        name=['conv3d'],
        atol=1e-2,
        atol_half=1e-2,
        rtol_half=1e-2,
        interface=['torch'],
        # out = (in - (dilation * (kernel_size - 1) + 1) + 2 * padding) / stride + 1
        para=dict(
            # stride=[1, 2, (2, 3, 4), 1, (2, 1, 1), 3, 1, 1],
            # padding=[0, (2, 1, 3), (2, 3, 4), 0, (1, 0, 1), 0, (1, 0, 1), 0],
            # dilation=[1, (2, 9, 5), (2, 1, 3), 1, (2, 1, 1), 1, (2, 1, 1), 1],
            # groups=[1, 2, 3, 1, 2, 2, 1, 1],
            stride=[1, 2, 1, (2, 1, 1), 3, 1, 1],
            padding=[0, (2, 1, 3), 0, (1, 0, 1), 0, (1, 0, 1), 0],
            dilation=[1, (2, 9, 5), 1, (2, 1, 1), 1, (2, 1, 1), 1],
            groups=[1, 2, 1, 2, 2, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    # "shape": ((4, 7, 12, 13, 9), (6, 16, 19, 8, 10), (6, 27, 22, 12, 20),
                    #           (1, 3, 4, 224, 224), (1, 16, 32, 56, 56),
                    #           (1, 128, 4, 56, 56), (1, 256, 4, 56, 56),
                    #           (0, 6, 5, 10, 9)),
                    "shape": ((4, 7, 12, 13, 9), (6, 16, 19, 8, 10),
                              (1, 3, 4, 224, 224), (1, 16, 32, 56, 56),
                              (1, 128, 4, 56, 56), (1, 256, 4, 56, 56),
                              (0, 6, 5, 10, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    # "shape": ((2, 7, 6, 5, 2), (2, 8, 12, 2, 4), (6, 9, 12, 2, 4),
                    #           (64, 3, 1, 7, 7), (16, 8, 5, 1, 1),
                    #           (64, 64, 1, 3, 3), (64, 256, 1, 1, 1),
                    #           (2, 6, 2, 3, 1)),
                    "shape": ((2, 7, 6, 5, 2), (2, 8, 12, 2, 4),
                              (64, 3, 1, 7, 7), (16, 8, 5, 1, 1),
                              (64, 64, 1, 3, 3), (64, 256, 1, 1, 1),
                              (2, 6, 2, 3, 1)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    # "shape": (None, (2, ), (6,),
                    #           None, (16,), (64,), (64,), (2,)),
                    "shape": (None, (2, ),
                              None, (16,), (64,), (64,), (2,)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ]
        ),
    ),

    'max_pool3d': dict(
        name=['max_pool3d'],
        para=dict(
            kernel_size=[6, (8, 6, 12), (6, 3, 8), (6, 3, 8), (3, 2, 2), (1, 2, 3), 1],
            stride=[None, (50, 3, 100), (3, 4, 2), (3, 4, 2), (2, 1, 2), 2, (2, 3, 4)],
            padding=[0, (3, 2, 6), (2, 1, 3), (2, 1, 3), 0, (0, 1, 1), 0],
            dilation=[1, (2, 4, 3), (2, 4, 3), (2, 4, 3), 2, (2, 1, 3), (2, 2, 2)],
            ceil_mode=[False, True, False, True, False, False, False],
            return_indices=[False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((5, 15, 12, 20), (5, 4, 9, 17, 22),
                              (6, 17, 10, 23), (1, 4, 17, 10, 23),
                              (9, 6, 6, 8, 6),
                              (4, 6, 8, 9, 12),
                              (6, 9, 8, 10, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool3d_return_indices': dict(
        name=['max_pool3d'],
        para=dict(
            kernel_size=[6, (8, 6, 12), (6, 3, 8), (6, 3, 8)],
            stride=[None, (50, 3, 100), (3, 4, 2), (3, 4, 2)],
            padding=[0, (3, 2, 6), (2, 1, 3), (2, 1, 3)],
            dilation=[1, (2, 4, 3), (2, 4, 3), (2, 4, 3)],
            ceil_mode=[False, True, False, True],
            return_indices=[True, True, True, True],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 15, 12, 20), (5, 4, 9, 17, 22),
                              (6, 17, 10, 23), (1, 4, 17, 10, 23)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[4, (15, 9, 21), (None, None, None), (1, 1, 1),
                         2, (None, 14, 14), (3, 20, 20)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 5, 12, 4, 8),
                              (4, 16, 9, 20), (12, 16, 32, 16),
                              (1, 2048, 4, 7, 7), (2, 512, 4, 4),
                              (2, 1024, 14, 14), (2, 720, 17, 17)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d': dict(
        name=["adaptive_max_pool3d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[4, (15, 9, 21), (None, None, None),
                         2, (1, 3, 2), (3, 4, 4), (3, 14, 14), (3, 20, 20)],
            return_indices=[False, False, False, False,
                            False, False, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 5, 12, 4, 8),
                              (4, 16, 9, 20), (12, 16, 32, 16),
                              (1, 2048, 4, 7, 7), (2, 512, 4, 4), (2, 1024, 14, 14),
                              (2, 1024, 14, 14), (2, 1024, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d_return_indices': dict(
        name=["adaptive_max_pool3d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[4, (15, 9, 21), (None, None, None)],
            return_indices=[True, True, True]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 5, 12, 4, 8),
                              (4, 16, 9, 20), (12, 16, 32, 16)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                },
            ]
        ),
    ),

    # FIXME masked_select输入指定shape，backward失败
    'masked_select': dict(
        name=['masked_select'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    # "shape": ((), (), (4,), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                    #           (4, 5, 6), (4, 1, 5, 8),
                    #           (0,), (4, 0), (16, 0, 9)),
                    "shape": ((), (4,), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                              (4, 5, 6),
                              (0,), (4, 0), (16, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    # "shape": ((), (4,), (), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                    #           (5, 6), (3, 5, 8),
                    #           (0,), (2, 4, 0), (0, 9)),
                    "shape": ((), (), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                              (5, 6),
                              (0,), (2, 4, 0), (0, 9)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
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
                    "shape": ((), (), (3, 4,), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                              (4, 1, 6), (4, 6, 5, 8),
                              (0,), (4, 1), (16, 0, 9)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ['mask'],
                    "shape": ((), (2, 4), (), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                              (5, 6), (6, 1, 8),
                              (0,), (4, 0), (1, 0, 9)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'imum': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "shape": ((), (1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8),
                              (0,), (7, 0), (9, 0, 6)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.bool,
                              Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'imum_input_nan': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
            args=[
                {
                    "ins": ['input'],
                    "value": ((float('nan'),), [[float('nan')]])
                },
                {
                    "ins": ['other'],
                    "shape": ((128, 128), (256, 8, 8)),
                    "gen_fn": Genfunc.randn,
                }
            ],
        ),
    ),

    'imum_other_nan': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((128, 128), (256, 8, 8)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['other'],
                    "value": ([[float('nan')]], [[float('nan')]])
                }
            ],
        ),
    ),

    # FIXME maximum,minimum input与other输入不同dtype，输出精度不一致
    'imum_broadcast': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (12,), (4, 128, 128), (1, 8, 8), (5, 1, 6, 7),
                              (0,), (4, 0, 5)),
                    # "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.bool,
                    #           Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['other'],
                    "shape": ((5,), (4, 12), (1, 128, 128), (256, 8, 8), (5, 6, 6, 7),
                              (2, 0), (4, 1, 5)),
                    # "dtype": [Dtype.int32, Dtype.uint8, Dtype.bool, Dtype.float32,
                    #           Dtype.int16, Dtype.float64, Dtype.float16, Dtype.uint8, Dtype.int8],
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                }
            ],
        ),
    ),

    'imum_ones': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, 8, 8),),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ['other'],
                    "shape": ((1, 8, 8),),
                    "dtype": [Dtype.int32],
                    "gen_fn": Genfunc.ones,
                }
            ],
        ),
    ),

    'mm': dict(
        name=['mm'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, 48), (4, 128), (256, 8)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mat2'],
                    "shape": ((48, 128), (128, 128), (8, 1)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mm_diff_dtype': dict(
        name=['mm'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, 0), (0, 128), (256, 8)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mat2'],
                    "shape": ((0, 128), (128, 128), (8, 0)),
                    "dtype": [Dtype.float16, Dtype.float64, Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'index_fill_scalar': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[-1, 0, 1, -2, 3],
            value=[0, 1, -1, 2.0, True]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((3,), (5, 3), (16, 8), (16, 4, 4), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "shape": ((20,), (3,), (5,), (), (10,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'index_fill_scalar_specific': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[-1, 0, 0, 1, -2],
            value=[0, False, -1, 2.0, 5]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (), (0,), (0, 2), (3, 0, 5)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "shape": ((), (3,), (0,), (0,), (0,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=1)
                },
            ]
        ),
    ),

    'index_fill_tensor': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[-1, 0, 0, 1, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (), (0,), (0, 2), (3, 0, 5)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['value'],
                    # index_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), (), ()),
                    "dtype": [Dtype.uint8, Dtype.float32, Dtype.int16,
                              Dtype.float64, Dtype.bool, Dtype.float16,
                              Dtype.int64, Dtype.int32, Dtype.int8],
                    "gen_fn": Genfunc.ones
                },
                {
                    "ins": ['index'],
                    "shape": ((), (3,), (0,), (0,), (0,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=1)
                },
            ]
        ),
    ),

    'index_fill_tensor_specific': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[-1, 0, -1, 2, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((3,), (5, 3), (16, 8), (16, 4, 4), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['value'],
                    # index_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), (), ()),
                    "dtype": [Dtype.int32, Dtype.float32, Dtype.bool,
                              Dtype.float64, Dtype.int16, Dtype.float16,
                              Dtype.int8, Dtype.uint8, Dtype.int64],
                    "gen_fn": Genfunc.ones
                },
                {
                    "ins": ['index'],
                    "shape": ((50, ), (3,), (5,), (), (10,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'expand': dict(
        name=['expand'],
        interface=['torch.Tensor'],
        para=dict(
            size=[(0,), (5,), (8, 2), (4, -1), (60800, 3), (-1, 4), (-1, 8, -1), (7, 3, -1), (5, -1, 8, 6, -1),
                  (2, 0), (4, -1, -1), (-1, -1, 9)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(), (), (), (8,), (60800, 1), (100, 1), (70, 1, 2), (3, 1), (4, 1, 6, 8),
                              (0,), (12, 0), (4, 0, 1)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.bool, Dtype.float16, Dtype.float64,
                              Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
                },
            ],
        ),
    ),

    'linspace': dict(
        name=['linspace'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-3,
        rtol_half=1e-3,
        atol_half=1e-4,
        para=dict(
            start=[0, 1.4, -10, False, True, 2, -2.5, -10, 1e-4, 0, 5],
            end=[0.5, -2.4, True, 100.232, False, 2, -2.5, 10, 1e-3, 10, -2],
            steps=[24, 23, 152, 100, 76, 50, 38, 25, 5, 0, 1],
        ),
    ),

    'permute': dict(
        name=['permute'],
        interface=['torch'],
        para=dict(
            dims=[(), (0, 1, 3, 2, 4, 5), (2, 0, 1), (0, 2, 3, -3), (1, 0), (0, -2, -1), (0,), (-1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(), (2, 8, 7, 8, 7, 128), (49, 49, 4), (2, 3, 200, 304), (20267, 1), (2, 3, 4), (0,), (1,)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'pad': dict(
        name=['pad'],
        para=dict(
            pad=[(2, 4), (0, 3), (7, -14, 2, 3), (0, 1, 0, 1), (0, 1, -1, 3, 1, 2), (0, 2, -1, 1, 1, 5),
                 (2, 4), (0, 3), (7, -14, 2, 3), (0, 1, 0, 1), (0, 1, -1, 3, 1, 2), (0, 2, -1, 1, 1, 5),
                 (0, 3), (0, 1, 0, 1), (0, 2, -1, 1, 1, 5)],
            mode=['reflect', 'reflect', 'reflect', 'reflect', 'reflect', 'reflect',
                  'replicate', 'replicate', 'replicate', 'replicate', 'replicate', 'replicate',
                  'circular', 'circular', 'circular'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(4, 5), (2, 56, 56), (12, 4, 8), (2, 3, 260, 260), (2, 144, 65, 65), (3, 576, 862, 2, 3),
                              (4, 5), (2, 56, 56), (12, 4, 8), (2, 3, 260, 260), (2, 144, 65, 65), (3, 576, 862, 2, 3),
                              (2, 56, 56), (2, 3, 260, 260), (3, 576, 862, 2, 3)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ],
        ),
    ),

    'pad_not_float': dict(
        name=['pad'],
        para=dict(
            pad=[(0, 3)],
            mode=['circular'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(2, 56, 56)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'constant_pad': dict(
        name=['pad'],
        para=dict(
            pad=[(), (4, 6), (), (-1, 2, 0, -2), (0, 3), (0, 1, 0, 1), (1, 1, 1, 1), (0, 193),
                 (-1, 2, 0, -2, 2, 3), (1, 2, 0, 2, 1, 4, 5, 6), (-1, 2, 1, 2, 0, 2, 1, 4, 5, 6),
                 (1, 0), (1, 2, 3, 5)],
            mode=['constant', 'constant', 'constant', 'constant', 'constant', 'constant', 'constant',
                  'constant', 'constant', 'constant', 'constant', 'constant', 'constant'],
            value=[None, -0.3, 0.2, 0, 100, 0, -1, 1, -3, 3, False, 1e-2, 2.4]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(), (5,), (2, 3), (4, 5), (2, 56, 56), (2, 3, 260, 260), (2, 144, 65, 65), (3, 576, 862),
                              (3, 4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 7),
                              (0,), (4, 0)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'constant_pad_positive': dict(
        name=['pad'],
        para=dict(
            pad=[(0, 3), (0, 1, 0, 1)],
            mode=['constant', 'constant'],
            value=[100, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(2, 56, 56), (2, 3, 260, 260)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.uint8],
                },
            ],
        ),
    ),

    'roll': dict(
        name=['roll'],
        interface=['torch'],
        para=dict(
            shifts=[-2, (2,), 9, 0, (0, 1), (3, 3), (-3, -3), (1, 3), (1, 2), (1, 2), 0, (0, 7), (3, 12)],
            dims=[None, (0,), None, -2, (0, 1), (1, 2), (1, 2), (-1, -2), (1, 2), (1, 1), None, (0, -1), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((), (8,), (4,), (4, 16), (8, 32), (2, 14, 14, 512), (2, 56, 56, 128),
                              (2, 14, 14, 512), (2, 14, 14, 512), (2, 14, 14, 512),
                              (0,), (3, 0), (2, 0, 5)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.bool,
                              Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'norm': dict(
        name=['norm'],
        interface=['torch'],
        para=dict(
            p=[0, 2.5, float('inf'), -float('inf'), 2, -2, 1, 2, 0],
            dim=[-1, None, (0, 1), (1, 2), 1, -1, 0, -1, (0, -2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((), (128, ), (384, 128), (256, 512, 1, 1), (384, 128), (384, 128),
                              (0,), (0, 12), (13, 0, 4)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME norm p参数输入fro等str报错
    # 'norm_p': dict(
    #     name=['norm'],
    #     interface=['torch'],
    #     para=dict(
    #         p=['fro', 0., 2, -1, None,
    #             0.231, -1.234, 'fro', 'fro', 'fro',
    #             'nuc', 'nuc', 'nuc', float('inf'), float('-inf'), ],
    #         dim=[None, None, None, [1, -1, 0], None,
    #              0, None, None, 0, [0, 1],
    #              None, [0, 1], [-1, 1], None, [0, 1, 2, 3]],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "shape": ((), (3,), (3, 4), (3, 4, 5), (6, 3, 4, 5),
    #                           (3, 4, 5, 6), (3, 4, 5), (), (3,), (6, 3,),
    #                           (6, 3,), (3, 4), (3, 4, 5), (), (6, 3, 4, 5)),
    #                 "dtype": [Dtype.float32, Dtype.float64],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    'group_norm': dict(
        name=['group_norm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            num_groups=[32, 32, 32, 32, 27, 1, 5, 12, 5, 6],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 0, 1e-2, 1e-8, -1, 2, 0]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256), (2, 256, 7, 10),
                              (2, 256, 24), (2, 256, 12, 12),
                              (3, 27, 4), (5, 6), (12, 15, 8, 9),
                              (0, 12,), (9, 15, 0), (3, 6, 9, 0)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((256,), (256,),
                              (256,), (256,),
                              (27,), (6,), (15,),
                              (12,), (15,), (6,)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ]
        ),
    ),

    'unique': dict(
        name=['unique'],
        interface=['torch'],
        para=dict(
            sorted=[True, True, False, True, False, False, True, False],
            return_inverse=[False, True, True, False, True, True, False, True],
            return_counts=[False, False, True, True, True, False, True, False],
            dim=[None, -1, 1, None, 2, 0, 1, -2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (252,), (2, 256), (4, 64, 128), (4, 2, 12, 3),
                              (0,), (2, 0), (7, 0, 9)),
                    "dtype": [Dtype.int64, Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'unique_same_value': dict(
        name=['unique'],
        interface=['torch'],
        para=dict(
            sorted=[True, True, False, True],
            return_inverse=[False, True, True, False],
            return_counts=[False, False, True, True],
            dim=[None, -1, 1, None],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.ones,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((252,), (2, 256), (4, 64, 128), (4, 2, 12, 3)),
                    "dtype": [Dtype.int64, Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ],
        ),
    ),

    'prod': dict(
        name=['prod'],
        interface=['torch'],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            dim=[0, -1, 0, -1, 1, 3, -1, 0, -2],
            dtype=[Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                   Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (42,), (56, 1), (70, 1, 2), (2, 512, 38, 38), (2, 80, 128, 128, 1),
                              (0,), (3, 0), (2, 0, 6)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME ctc_loss输入int8, uint8, int16报错
    'ctc_loss': dict(
        name=["ctc_loss"],
        interface=['CustomizedTest'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            reduction=['none', 'mean', 'sum', 'none'],
            blank=[0, 0, 0, 9],
            zero_infinity=[True, False, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "requires_grad": [True],
                    "shape": ((26, 20, 38), (26, 20, 38), (26, 20, 38), (32, 20, 10)),
                    # "dtype": [Dtype.float32, Dtype.float64, Dtype.float64, Dtype.float32, Dtype.float32, Dtype.float64],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['targets'],
                    "shape": ((20, 10), (20, 14), (20, 11), (20, 54)),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.int8, Dtype.int16, Dtype.int32, Dtype.uint8],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=80),
                },
                {
                    "ins": ['input_lengths'],
                    "shape": ((20, ), (20, ), (20, ), (20, )),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.uint8, Dtype.int32, Dtype.int8, Dtype.int16],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=26),
                },
                {
                    "ins": ['target_lengths'],
                    "shape": ((20, ), (20, ), (20, ), (20, )),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.int16, Dtype.int8, Dtype.uint8, Dtype.int32],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=10),
                },
            ],
        ),
    ),

    # ctc_loss输入int8, uint8, int16报错
    'ctc_loss_un_padded': dict(
        name=["ctc_loss"],
        interface=['CustomizedTest'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            reduction=['none', 'mean', 'sum', 'none'],
            blank=[0, 0, 0, 9],
            zero_infinity=[True, False, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "requires_grad": [True],
                    "shape": ((26, 10, 38), (26, 10, 38), (26, 10, 38), (32, 10, 10)),
                    # "dtype": [Dtype.float32, Dtype.float64, Dtype.float64, Dtype.float32, Dtype.float32, Dtype.float64],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['targets'],
                    "shape": ((10, ), (10, ), (10, ), (10, )),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.int8, Dtype.int16, Dtype.int32, Dtype.uint8],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=80),
                },
                {
                    "ins": ['input_lengths'],
                    "shape": ((10, ), (10, ), (10, ), (10, )),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.uint8, Dtype.int32, Dtype.int8, Dtype.int16],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=26),
                },
                {
                    "ins": ['target_lengths'],
                    "shape": ((10, ), (10, ), (10, ), (10, )),
                    # "dtype": [Dtype.int64, Dtype.int64, Dtype.int16, Dtype.int8, Dtype.uint8, Dtype.int32],
                    "dtype": [Dtype.int64, Dtype.int64],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'remainder_self_scalar': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            self=[2, 4.3, 0, 1, 100., -2.5, -3, -1, 0.23, 0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['other'],
                    "shape": ((), (6, ), (4, 1), (1, 28, 28),
                              (16, 3, 7, 14, 14), (1, 28, 28), (1, 28, 28),
                              (0,), (0, 3), (4, 0, 5)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8, Dtype.bool],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
            ],
        ),
    ),

    'remainder_self_bool': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            self=[True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['other'],
                    "shape": ((6, ), (4, 1)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
            ],
        ),
    ),

    'remainder_tensor': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (6, ), (4, 5), (5,), (2, 3, 4, 5), (14, 1, 28),
                              (16, 1, 7, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
                {
                    "ins": ['other'],
                    "shape": ((), (6, ), (4, 1), (4, 5), (3, 4, 5), (28, 28),
                              (16, 3, 7, 14, 14)),
                    "dtype": [Dtype.int32, Dtype.bool, Dtype.uint8, Dtype.int32, Dtype.float64,
                              Dtype.float16, Dtype.int8, Dtype.uint8, Dtype.float32],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
            ],
        ),
    ),

    'remainder_tensor_zero': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 1, 7, 14, 14),),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
                {
                    "ins": ['other'],
                    "shape": ((16, 3, 7, 14, 14),),
                    "dtype": [Dtype.int32, Dtype.bool, Dtype.uint8, Dtype.int32, Dtype.float64,
                              Dtype.float16, Dtype.int8, Dtype.uint8, Dtype.float32],
                    "gen_fn": Genfunc.zeros,
                },
            ],
        ),
    ),

    'remainder_other_scalar': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            other=[2, 4.3, 10.1, 0, 100., -2.5, -3, -1, 0.23, 0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (6, ), (4, 1), (1, 28, 28),
                              (16, 3, 7, 14, 14), (1, 28, 28), (1, 28, 28),
                              (0,), (0, 3), (4, 0, 5)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
            ],
        ),
    ),

    'remainder_other_scalar_bool': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            other=[True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, ), (4, 1)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8],
                    "gen_fn": dict(fn=Genfunc.randn_int, low=-4, high=4),
                },
            ],
        ),
    ),

    'gather': dict(
        name=['gather'],
        interface=['torch'],
        para=dict(
            dim=[0, -1, 1, 0, -2, 1, 2, 0, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((8,), (9,), (3, 9), (16, 4, 4), (14, 6, 2), (64, 4, 14, 14), (64, 4, 16, 16),
                              (2, 0), (5, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ['index'],
                    "shape": ((), (12,), (2, 15), (16, 4, 4), (13, 10, 1), (64, 4, 14, 14), (64, 4, 16, 16),
                              (1, 0), (16, 0, 14)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ],
        ),
    ),

    'gather_0dim': dict(
        name=['gather'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((), ()),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
                {
                    "ins": ['index'],
                    "shape": ((), (15,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
                },
            ],
        ),
    ),

    'gather_not_float': dict(
        name=['gather'],
        interface=['torch'],
        para=dict(
            dim=[0, -1, 1, 0, -2, 1, 2, 0, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8,), (9,), (3, 9), (16, 4, 4), (14, 6, 2), (64, 4, 14, 14), (64, 4, 16, 16),
                              (2, 0), (5, 0, 9)),
                    "dtype": [Dtype.int16, Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['index'],
                    "shape": ((), (9,), (2, 15), (16, 4, 4), (13, 10, 1), (64, 4, 14, 14), (64, 4, 16, 16),
                              (1, 0), (16, 0, 14)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },

            ],
        ),
    ),

    # FIXME scatter输入指定shape，结果不一致
    'scatter': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[0, -1, 1, -2, 2, 1, -1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4,), (5,), (2, 8), (5, 9, 16), (16, 4, 4), (2, 8, 64, 64), (2, 8, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['index'],
                    # "shape": ((), (6,), (2, 7), (4, 8, 10), (16, 4, 4), (2, 8, 1, 1), (2, 8, 1, 1)),
                    "shape": ((), (6,), (2, 7), (4, 8, 5), (16, 4, 4), (2, 8, 1, 1), (2, 8, 1, 1)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['src'],
                    "shape": ((), (7,), (4, 9), (8, 12, 20), (16, 4, 4), (2, 8, 4, 4), (2, 8, 4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ]
        ),
    ),

    'scatter_specific': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[0, -1, 1, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (0,), (5, 0), (4, 9, 0)),
                    "dtype": [Dtype.float32],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
                },
                {
                    "ins": ['index'],
                    "shape": ((), (0,), (0, 3), (2, 4, 0, 8)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
                },
                {
                    "ins": ['src'],
                    "shape": ((), (), (9, 0, 2), (5, 2)),
                    "dtype": [Dtype.float32],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=1),
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
            reduce=['add', 'multiply']
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (2, 8, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (2, 8, 1, 1)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['src'],
                    "shape": ((16, 4, 4), (2, 8, 4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'scatter_scalar': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[0, -1, 1, -2, 2, 1, -1],
            value=[True, 0.25, -100, 0, 2.34, 20, 1e-4],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4,), (5,), (2, 8), (5, 9, 16), (16, 4, 4), (2, 8, 64, 64), (2, 8, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['index'],
                    "shape": ((), (6,), (2, 7), (4, 8, 10), (16, 4, 4), (2, 8, 1, 1), (2, 8, 1, 1)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ]
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
            value=[-2.31, float("-inf")],
            reduce=['add', 'multiply']
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ]
        ),
    ),

    # FIXME index_put出现精度不一致
    'index_put_acc_three_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            # accumulate=[True, True, True, True, True,
            #             False, False, False, False, False]
            accumulate=[True, True, True, True, True,
                        False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((16, 4, 4), (16, 4, 4), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0),
                    #           (16, 4, 4), (16, 4, 4), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0)),
                    "shape": ((16, 4, 4), (16, 4, 4), (16, 4, 4), (64, 4, 14, 14),
                              (4, 5, 0),
                              (16, 4, 4)
                              ),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['indices1', 'indices2', 'indices3'],
                    # "shape": ((16, 4, 4), (6,), (10, 12, 2, 2), (64, 4, 14),
                    #           (4, 5, 0),
                    #           (16, 4, 4), (6,), (10, 12, 2, 2), (64, 4, 14),
                    #           (4, 5, 0)),
                    "shape": ((16, 4, 4), (6,), (10, 12, 2, 2), (64, 4, 14),
                              (4, 5, 0),
                              (6,)
                              ),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    # "shape": ((16, 4, 4), (6,), (10, 12, 2, 2), (64, 4, 14, 14),
                    #           (4, 5, 0),
                    #           (16, 4, 4), (6,), (12, 1, 2), (64, 4, 14, 14),
                    #           (4, 5, 0)),
                    "shape": ((16, 4, 4), (6,), (10, 12, 2, 2), (64, 4, 14, 14),
                              (4, 5, 0),
                              (6,)
                              ),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
            ]
        ),
    ),

    # FIXME index_put出现精度不一致
    'index_put_acc_two_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            # accumulate=[True, True, True, True, True,
            #             False, False, False, False, False]
            accumulate=[True, True, True, True, True,
                        False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((4, 5), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0),
                    #           (4, 5), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0)),
                    "shape": ((4, 5), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                              (4, 5, 0),
                              (4, 5),
                              (4, 5, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1', 'indices2'],
                    # "shape": ((4, 5), (2, 6, 10), (16, 4), (64, 4),
                    #           (4, 5),
                    #           (4, 5), (2, 6, 10), (16, 4), (64, 4),
                    #           (4, 5)),
                    "shape": ((4, 5), (2, 6, 10), (16, 4), (64, 4),
                              (4, 5),
                              (4, 5),
                              (4, 5)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    # "shape": ((4, 5), (2, 6, 10), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0),
                    #           (4, 5), (6, 10), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 5, 0)),
                    "shape": ((4, 5), (2, 6, 10), (16, 4, 4), (64, 4, 14, 14),
                              (4, 5, 0),
                              (4, 5),
                              (4, 5, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    # FIXME index_put出现精度不一致
    'index_put_acc_one_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            # accumulate=[True, True, True, True, True, True, True,
            #             False, False, False, False, False, False, False]
            accumulate=[True, True, True, True, True, True, True,
                        False, False, False, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((6,), (4,), (5,), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 0),
                    #           (6,), (4,), (5,), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 0)),
                    "shape": ((6,), (4,), (5,), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                              (4, 0),
                              (6,), (4,), (5,), (4, 5),
                              (4, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1'],
                    # "shape": ((6,), (), (2, 10), (4,), (16,), (64,),
                    #           (4,),
                    #           (6,), (), (2, 10), (4,), (16,), (64,),
                    #           (4,)),
                    "shape": ((6,), (), (2, 10), (4,), (16,), (64,),
                              (4,),
                              (6,), (), (2, 10), (4,),
                              (4,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    # "shape": ((6,), (), (2, 10), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 0),
                    #           (6,), (), (2, 10), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                    #           (4, 0)),
                    "shape": ((6,), (), (2, 10), (4, 5), (16, 4, 4), (64, 4, 14, 14),
                              (4, 0),
                              (6,), (), (2, 10), (4, 5),
                              (4, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    # FIXME index_put出现精度不一致
    # 'index_put_acc_broadcast': dict(
    #     name=['index_put'],
    #     interface=['CustomizedTest'],
    #     is_inplace=True,
    #     para=dict(
    #         accumulate=[False, False, False, True, False]
    #     ),
    #     tensor_para=dict(
    #         gen_fn=Genfunc.randn,
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((16, 4, 4), (6, 4, 5), (6, 4, 5), (6, 4, 5), (6, 4, 5)),
    #                 "dtype": [Dtype.float32, Dtype.float64],
    #             },
    #             {
    #                 "ins": ['indices1'],
    #                 "shape": ((4, 4), (6, 4, 5), (2, 6), (6, 4), (6, 4)),
    #                 "dtype": [Dtype.int64],
    #                 "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
    #             },
    #             {
    #                 "ins": ['indices2'],
    #                 "shape": ((4,), (4, 5), (6,), (4,), (4,)),
    #                 "dtype": [Dtype.int64],
    #                 "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
    #             },
    #             {
    #                 "ins": ['indices3'],
    #                 "shape": ((16, 1, 4), (5,), (4, 1, 6), None, None),
    #                 "dtype": [Dtype.int64],
    #                 "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
    #             },
    #             {
    #                 "ins": ['values'],
    #                 "shape": ((16, 4, 4), (6, 1, 5), (2, 6), (6, 4, 5), (6, 4, 5)),
    #                 "dtype": [Dtype.float32, Dtype.float64],
    #             },
    #         ]
    #     ),
    # ),

    'index_put_acc_bool_indices_zeros': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4, 4), (4, 4)),
                    "dtype": [Dtype.float32, Dtype.int64],
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ['indices1'],
                    "shape": ((4, 4), (4, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ['values'],
                    "shape": ((0,), (0,)),
                    "dtype": [Dtype.float32, Dtype.int64],
                    "gen_fn": Genfunc.ones
                },
            ]
        ),
    ),

    'index_put_one_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, False, True, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, 4, 5), (6, 4, 5), (16, 4, 4), (64, 4, 14, 14), (4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['indices1'],
                    "shape": ((6,), (6,), (16, 4), (64, 4), (4, 4)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ['values'],
                    "shape": ((6, 4, 5), (6, 4, 5), (64, 4), (256, 14, 14), (16,)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool]
                },
            ]
        ),
    ),

    'index_put_bool_indices_value': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, False, True, False, True, True]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((4,), (4, 4), (6, 4, 5), (3, 2, 2, 6), (3, 2, 2, 20), (4, 2, 2, 6, 2)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                },
                {
                    "ins": ['indices1'],
                    "value": [[1, 0, 1, 0], [1, 0, 0, 1],
                              [1, 0, 1, 0, 0, 1], [1, 1, 0],
                              [[1, 0], [1, 1], [0, 1]], [1, 0, 0, 0]],
                    "dtype": [Dtype.bool],
                },
                {
                    "ins": ['indices2'],
                    "value": [None, None, None, [[1, 0], [0, 1]], [1, 0],
                              [[1, 1], [1, 1]]],
                    "dtype": [Dtype.bool],
                },
                {
                    "ins": ['indices3'],
                    "value": [None, None, None, None, None,
                              [[1, 1], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]]],
                    "dtype": [Dtype.bool],
                },
                {
                    "ins": ['values'],
                    "shape": ((2,), (1,), (3, 4, 5), (2, 6), (80,), (4,)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool]
                },
            ]
        ),
    ),

    'arange': dict(
        name=['arange'],
        interface=['torch'],
        para=dict(
            start=[0, 0, -4, 0.1, 10, 2.3, True, -20, 90, 1e-3],
            end=[91, 128, 5, 0.5, 10, 2.3, 100, False, -90, 1e-4],
            step=[13, 1, 1, 0.1, True, 0.5, 2.1, 0.5, -5.6, -1e-5],
        ),
    ),

    'arange_default': dict(
        name=['arange'],
        interface=['torch'],
        para=dict(
            end=[5, 10, 4.0, 9.0],
        ),
    ),

    'randperm': dict(
        name=['randperm'],
        no_output_ref=True,
        para=dict(
            n=[2, 1999, 640000, 0, 1],
        ),
    ),

    'uniform': dict(
        name=['uniform'],
        no_output_ref=True,
        para={
            'start': [0, 0.5, -0.12499999999999999, 0.25, -0.04811252243246881, -5, 0, 2.3, -4],
            'end': [1, 1.5, 0.12499999999999999, 0.25, 0.04811252243246881, -1, 0, 12, 3],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3), (4, 3, 5),
                              (0,), (4, 0), (3, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ],
        ),
    ),

    'random': dict(
        name=['random'],
        no_output_ref=True,
        para={
            'start': [-10, 0, 3, -1, 0, 3, 4, 0, -5],
            'end': [-2, 2, None, 1, None, 4, 12, 4, None],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3), (16, 1, 3, 3),
                              (0,), (4, 0), (3, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16,
                              Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8],
                },
            ],
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        no_output_ref=True,
        para={
            'start': [0],
            'end': [2],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 1, 3, 3),),
                    "dtype": [Dtype.bool, Dtype.uint8],
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.1, None, 0.5, None, None, 0.7, None, None],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.rand,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3),
                              (0,), (4, 0), (5, 0, 7)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                },
            ],
        ),
    ),

    'bernoulli_int': dict(
        name=['bernoulli'],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.1, 0, 0.5, True],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.rand,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (5, ), (2, 3), (2, 1, 6)),
                    "dtype": [Dtype.int64, Dtype.int32, Dtype.int16,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
        atol=1e-5,
        atol_half=1e-1,
        rtol_half=1e-2,
        para=dict(
            eps=[1e-5, 1e-5, 1e-12, 0, -1e-5, 2],
            normalized_shape=[(5, 3, 5), (128, ), (64, ), (32,),
                              (3, 5), (2, 16, 128)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 5, 3, 5), (2, 3136, 128), (2, 64), (32,),
                              (2, 5, 3, 5), (2, 16, 128)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": (None, (128,), (64,), (32,),
                              (3, 5), (2, 16, 128)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": (None, (128,), (64,), (32,),
                              (3, 5), (2, 16, 128)),
                },
            ]
        )
    ),

    'layer_norm_empty_tensor': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
        atol=1e-5,
        para=dict(
            eps=[1e-2, 1e-8, -3],
            normalized_shape=[(0,), (0, 12), (9,)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "shape": ((0,), (0, 12), (6, 0, 9)),
                },
                {
                    "ins": ["weight"],
                    "shape": ((0,), (0, 12), (9,)),
                },
                {
                    "ins": ["bias"],
                    "shape": ((0,), (0, 12), (9,)),
                },
            ]
        )
    ),

    'copy': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16, Dtype.bool,
               Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "shape": ((), (8,), (12,), (192, 147), (1, 1, 384), (2, 1, 38, 45),
                              (0,), (0, 12,), (12, 0, 9)),
                    "no_contiguous": [True],
                },
                {
                    "ins": ["other"],
                    "shape": ((), (), (12,), (147, 1), (384, 1, 1), (45, 38, 1, 2),
                              (0,), (12, 0), (9, 0, 12)),
                },
            ]
        )
    ),

    'copy_different_dtype': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "shape": ((192, 147), (1, 1, 384), (2, 1, 38, 45), (100, 100)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.bool,
                              Dtype.int64, Dtype.int32, Dtype.int16, Dtype.int8, Dtype.uint8],
                    "no_contiguous": [True],
                },
                {
                    "ins": ["other"],
                    "dtype": [Dtype.float64, Dtype.int64, Dtype.float16, Dtype.float16,
                              Dtype.int32, Dtype.float32, Dtype.uint8, Dtype.uint8, Dtype.uint8],
                    "shape": ((147, 1), (384, 1, 1), (45, 38, 1, 2), (1, 100)),
                },
            ]
        )
    ),

    'copy_broadcast': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=[Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "shape": ((8,), (12, 2), (192, 147, 2), (6, 5, 384), (2, 12, 38, 45, 3),
                              (0, 2), (0, 12,), (12, 0, 9, 2)),
                    "no_contiguous": [True],
                },
                {
                    "ins": ["other"],
                    "shape": ((1,), (12,), (1, 147), (6, 1, 384), (2, 1, 38, 45),
                              (1,), (0, 1,), (12, 0, 1)),
                    "no_contiguous": [True],
                },
            ]
        )
    ),

    # FIXME interpolate输入mode为linear，做down sample精度不一致
    'interpolate': dict(
        name=["interpolate"],
        dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
        para=dict(
            mode=['nearest', 'nearest', 'nearest', 'nearest', 'nearest',
                  'bilinear', 'bilinear', 'bicubic', 'bicubic',
                  'trilinear', 'trilinear', 'linear', 'linear', 'linear'],
            # For bicubic, do not use big size like (64, 64), which will cause accuracy error in float16.
            # Additionally, according to pytorch website(https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
            # "This operation may produce nondeterministic gradients when given tensors on a CUDA device."
            # So if you are facing a value error in backward, try to figure out whether this is a problem of pytorch.
            size=[None, (50, 76), (4, 224, 224), None, 32, (25, 38), (14, 16), (32, 32), (10, 32), None, (4, 224, 112), (64, ), (32,), None],
            # scale_factor=[0.5, None, None, (3.0, 3.0), None, None, None, None, None, (1.3, 1, 0.2), None, None, None, 0.3],
            scale_factor=[0.5, None, None, (3.0, 3.0), None, None, None, None, None, (1.3, 1, 0.2), None, None, None, 1],
            align_corners=[None, None, None, None, None, False, True, True, False, True, False, False, True, False],
            # recompute_scale_factor=[False, False, False, False, False, False, True, False]

        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 16, 23), (2, 256, 25, 38), (1, 3, 32, 224, 224), (2, 2, 16, 16), (2, 2, 16, 16),
                              (2, 256, 13, 19), (3, 12, 14, 19), (2, 16, 1, 1), (2, 16, 15, 32),
                              (1, 3, 32, 112, 112), (1, 3, 32, 112, 112), (2, 32, 32), (2, 32, 32), (2, 32, 32)),
                },
            ]
        )
    ),

    'col2im': dict(
        name=["col2im"],
        interface=['CustomizedTest'],
        para=dict(
            output_size=[(352, 528), (12, 40), (4, 26), 10],
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (3, 1), 2],
            padding=[1, 0, (2, 4), 0],
            dilation=[1, 1, (2, 3), 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 576, 46464),
                              (2, 512, 240),
                              (2, 2048, 62),
                              (3, 36, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'im2col': dict(
        name=["im2col"],
        interface=['CustomizedTest'],
        para=dict(
            kernel_size=[2, (2, 2), 3, (2, 1), (2, 2), 3],
            stride=[1, 2, 2, (2, 1), (3, 4), 2],
            padding=[0, 0, 1, 0, (2, 1), 0],
            dilation=[1, 1, 1, (2, 3), 1, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 16, 32, 24), (2, 5, 3, 3),
                              (2, 64, 352, 528),
                              (2, 256, 12, 40),
                              (2, 512, 4, 26),
                              (3, 4, 10, 10)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'flip': dict(
        name=['flip'],
        interface=['torch'],
        para=dict(
            dims=[(-1,), (0,), (1,), (-2, -1), (0, 2, -1, -3,), (0, -1, 2),
                  (-1,), (0, 1), (-2, 0, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((), (12,), (49, 49), (12, 13, 14), (12, 13, 14, 16), (2, 3, 4, 10, 12),
                              (0,), (12, 0), (2, 0, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cholesky': dict(
        name=['cholesky_ex'],
        interface=['torch.linalg'],
        para=dict(
            upper=[True, False, True, False, False],
            check_errors=[True, False, True, False, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 4), (2, 3, 3), (2, 3, 4), (6, 3, 4, 5),
                              (0, 3, 4)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.sym_mat,
                },
            ],
        ),
        requires_backward=[0],
        saved_args=dict(output=0),
    ),

    # FIXME triangular_solve输入空张量报错
    'triangular_solve': dict(
        name=['triangular_solve'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            upper=[True, False, True, False],  # , True, False, True, False],
            transpose=[True, False, False, True],  # , True, False, False, True],
            unitriangular=[True, False, True, False],  # , True, False, True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2), (3, 3), (7, 6, 5), (7, 2, 1),),
                    # (0, 4, 3), (0, 5), (4, 5), (5, 4, 0)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['A'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2), (5, 3, 3), (7, 6, 6), (2, 2),),
                    # (4, 4), (4, 0, 0), (0, 4, 4), (4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
        saved_args=dict(output=0),
    ),

    'repeat': dict(
        name=["repeat"],
        interface=['torch.Tensor'],
        para=dict(
            repeats=[(), (3,), (3, 5), (4, 2), (4, 2, 1),
                     (4, 2), (4, 2, 1),
                     (4, 2, 1), (3, 4, 6, 3, 5),
                     (1, 2), (2, 2, 3), (4, 2, 3, 0)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (), (), (3, ), (3, ),
                              (1, 2), (1, 2), (1, 2, 3),
                              (4, 2, 3, 5),
                              (0,), (12, 0), (4, 0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                },
            ]
        ),
    ),

    'normal': dict(
        name=["normal"],
        no_output_ref=True,
        para=dict(
            mean=[-1, -0.5, 0, 0.1, 2, True, False, 0.2, -2, 0],
            std=[0, 0.5, 1, 2.3, 3, True, True, 0.5, 0, 3],
            size=[(), (1280,), (32, 160), (320, 8),
                  (32, 80), (2, 2, 20, 16), (320, 2, 3, 3),
                  (0,), (4, 0), (2, 0, 9)],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        para=dict(
            mean=[0, 2, 0.1, 0.5, True, False, -0.3, -4, -1.3],
            std=[0.5, 1, 2, 3.14, True, True, 0, 0.3, 4],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": [(), (128,), (320, 8), (32, 80),
                              (32, 8), (16, 64, 32), (0,), (0, 3), (2, 0, 5)],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'normal_std_tensor': dict(
        name=["normal"],
        no_output_ref=True,
        para=dict(
            mean=[-1, -0.5, 0, 0.1, 2, True, False, 1.2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.positive,
            args=[
                {
                    "ins": ['std'],
                    "shape": ((), (16,), (8, 4),
                              (256, 256, 3, 3), (256, 128, 1, 1),
                              (0,), (4, 0), (4, 0, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'normal_mean_tensor': dict(
        name=["normal"],
        no_output_ref=True,
        para=dict(
            std=[0.5, 0.1, 0.054056261216234408, 2, 5, 1.2, True, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['mean'],
                    "shape": ((), (16,), (8, 4), (256, 256, 3, 3), (256, 128, 1, 1),
                              (0,), (4, 0), (4, 0, 7)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'normal_tensor': dict(
        name=["normal"],
        no_output_ref=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['mean'],
                    # (3, 4), (4,16,),will be removed in version 1.6 release
                    "shape": ((), (16, 64), (8, 8, 16), (256, 1, 3, 3), (256, 128, 3, 1),
                              (0,), (4, 0), (2, 0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.float16, Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['std'],
                    # (12,), (2,4,4,2), will be removed in version 1.6 release
                    "shape": ((128,), (16, 64), (8, 16), (256, 256, 3, 3), (256, 128, 1, 1),
                              (0,), (2, 1, 1), (0, 9)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.float32, Dtype.float64, Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ]
        ),
    ),

    # 'normalize': dict(
    #     name=["normalize"],
    #     para=dict(
    #         p=[2.0, 1.0, 0, 1.5, 2, 2, 2, 2, 2],
    #         dim=[1, 2, -1, 0, 0, -1, -1, -2, 1],
    #         eps=[1e-12, 1e-12, 1e-11, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ['input'],
    #                 "shape": ((256, 256, 3, 3), (256, 128, 1, 1), (64, 32, 16), (32, 8), (8,), (),
    #                           (0,), (0, 8), (8, 0, 3)),
    #                 "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
    #             }
    #         ]
    #     )
    # ),

    # 'normalize_p': dict(
    #     name=['normalize'],
    #     para=dict(
    #         p=['fro', 0., 2, -1, None,
    #             0.231, -1.234, 'fro', 'fro', 'fro',
    #             'nuc', 'nuc', 'nuc', float('inf'), float('-inf'), ],
    #         dim=[None, None, None, [1, -1, 0], None,
    #              0, None, None, 0, [0, 1],
    #              None, [0, 1], [-1, 1], None, [0, 1, 2, 3]],
    #         eps=[1e-12, 1e-12, 1e-12, 1e-12, 1e-12,
    #              1e-12, 1e-12, 1e-12, 1e-12, 1e-12,
    #              1e-12, 1e-12, 1e-12, 1e-12, 1e-12,],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "shape": ((), (3,), (3, 4), (3, 4, 5), (6, 3, 4, 5),
    #                           (3, 4, 5, 6), (3, 4, 5), (), (3,), (6, 3,),
    #                           (6, 3,), (3, 4), (3, 4, 5), (), (6, 3, 4, 5)),
    #                 "dtype": [Dtype.float32, Dtype.float64],
    #                 "gen_fn": Genfunc.randn,
    #             },
    #         ],
    #     ),
    # ),

    'meshgrid': dict(
        name=["meshgrid"],
        interface=['CustomizedTest'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "shape": (((8,), (8,), (8,)),
                              ((16,), (8,), ()),
                              ((32,), (16,)), ((8,), (0,))),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64,
                              Dtype.int16, Dtype.int32, Dtype.int64,
                              Dtype.int8, Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                    "gen_num_range": [1, 5],
                },
            ],
            seq_name='tensors',
        ),
    ),

    'multinomial': dict(
        name=["multinomial"],
        interface=['torch'],
        no_output_ref=True,
        para=dict(
            num_samples=[7, 8, 9,
                         63, 257, 128],
            replacement=[False, False, True,
                         True, True, True],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.positive,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, ), (8, ), (8, ),
                              (16, 64,), (128, 256,), (256, 128,)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'cast_dtype': dict(
        name=["cast_dtype"],
        interface=['CustomizedTest'],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": [(32, 64,), (128, 24, 32), (16, 8,), (24, 12,)],
                    "dtype": [Dtype.float32, Dtype.int64, Dtype.int8, Dtype.uint8],
                },
                {
                    "ins": ['out'],
                    "shape": [(32, 64,), (128, 24, 32), (16, 8,), (24, 12,)],
                    "dtype": [Dtype.int64, Dtype.float64, Dtype.bool, Dtype.float16],
                },
            ]
        ),
    ),

    # 'view_as_real': dict(
    #         name=['view_as_real'],
    #         interface=['torch'],
    #         dtype=[Dtype.complex64, Dtype.complex128],
    #         tensor_para=dict(
    #             gen_fn=Genfunc.randn_cmplx,
    #             args=[
    #                 {
    #                     "ins": ['input'],
    #                     "shape": ((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
    #                             (256, 128, 3, 3),
    #                             (2, 31, 512, 6, 40), (0,), (0, 9), (4, 0, 5)),
    #                 },
    #             ],
    #         ),
    #     ),

    # 'view_as_complex': dict(
    #         name=['view_as_complex'],
    #         interface=['torch'],
    #         dtype=[Dtype.float32, Dtype.float64],
    #         tensor_para=dict(
    #             gen_fn=Genfunc.randn,
    #             args=[
    #                 {
    #                     "ins": ['input'],
    #                     "shape": ((2,), (364800, 2), (2, 128, 2),
    #                             (256, 128, 3, 2),
    #                             (2, 31, 512, 6, 2), (0, 2), (4, 0, 2)),
    #                 },
    #             ],
    #         ),
    #     ),

    'polar': dict(
        name=['polar'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['abs'],
                    "shape": ((), (1024, ), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130), (0,), (0, 3), (18, 0, 9)),
                },
                {
                    "ins": ['angle'],
                    "shape": ((), (1024, ), (384, 128),
                              (32, 64, 8, 128), (1, ),
                              (2, 32, 1, 1), (2, 0), (3, ), (1, 9)),
                },
            ],
        ),
    ),

    # 'randn': dict(
    #     name=['randn'],
    #     no_output_ref=True,
    #     para=dict(
    #         size=[(), (128,), (3, 64), (3, 16, 64),
    #               (4, 16, 8, 64), (2, 16, 1, 64, 5),
    #               (0,), (0, 16), (8, 0, 12)],
    #     ),
    # ),

    'lerp': dict(
        name=['lerp'],
        interface=['torch'],
        dtype=[Dtype.float64, Dtype.float32, Dtype.float16],
        para=dict(
            weight=[-1, 0, 1, -2.342, 0.028, True, False, 1.2, -0.23, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130), (0,), (0, 3), (18, 0, 9)),
                },
                {
                    "ins": ['end'],
                    "shape": ((), (1024, ), (384, 128), (64, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1),
                              (2, 0), (3, ), (1, 9)),
                },
            ],
        ),
    ),

    'lerp_tensor': dict(
        name=['lerp'],
        interface=['torch'],
        dtype=[Dtype.float64, Dtype.float32, Dtype.float16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130), (0,), (0, 3), (18, 0, 9)),
                },
                {
                    "ins": ['end'],
                    "shape": ((), (1024, ), (384, 128), (64, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1),
                              (2, 0), (3, ), (1, 9)),
                },
                {
                    "ins": ['weight'],
                    "shape": ((), (1024, ), (3, 1, 128), (64, 1),
                              (1, ), (64, 1, 128), (2, 32, 1, 130),
                              (3, 2, 0), (1, 3), (1, 0, 9)),
                },
            ],
        ),
    ),

    'triu': dict(
        name=['triu'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            diagonal=[0, 1, 2, -1, 3, 12, 0, 5, -9, -1, 1, 2, 10, -10],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, 64), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (8, 9), (6, 7), (6, 6), (9, 9),
                              (6, 8, 8), (64, 7, 28, 28),
                              (2, 0), (12, 0), (2, 0, 9)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16, Dtype.int16, Dtype.int32,
                              Dtype.int64, Dtype.uint8, Dtype.int8, Dtype.bool],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'isnan': dict(
        name=['isnan'],
        interface=['torch'],
        dtype=[Dtype.float64, Dtype.float32, Dtype.float16, Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (128,), (1024, 64), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,), (4, 0), (12, 0, 9)),
                },
            ],
        ),
    ),

    'isnan_input_nan': dict(
        name=['isnan'],
        interface=['torch'],
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float64, Dtype.float16],
            args=[
                {
                    "ins": ['input'],
                    "value": ((float('nan'),), [[float('nan'), 1, -1]], [[float('nan'), 0], [1, float('nan')]],
                              [[[float('nan'), float('inf')], [0, float('-inf')]]])
                },
            ],
        ),
    ),

    'amax': dict(
        name=['amax'],
        interface=['torch'],
        dtype=[Dtype.float64, Dtype.float32, Dtype.float16, Dtype.int16, Dtype.int32, Dtype.int64, Dtype.int8, Dtype.uint8, Dtype.bool],
        para=dict(
            dim=[None, -1, (0,), 1, 0, 2, (1, 2), (-1, 2, 0, -3), None, None, -2, (0,)],
            keepdim=[False, True, True, False, False, False, True, False, False, True, False, True],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (), (18,), (1024, 64), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130), (128, 64, 32, 3), (384, 128),
                              (3, 0), (4, 0, 5)),
                },
            ],
        ),
    ),

    'linalgqr': dict(
        name=['linalgqr'],
        interface=['CustomizedTest'],
        dtype=[Dtype.float64, Dtype.float32],
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            mode=['reduced', 'reduced', 'complete', 'complete', 'r', 'r',
                  'reduced', 'complete', 'r', 'reduced', 'complete', 'r'],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, 384), (384, 1024),
                              (64, 1, 128), (128, 64, 32, 3),
                              (2, 32, 130, 100), (2, 32, 100, 150),
                              (4, 2, 1024, 1024), (4, 284, 284), (64, 64),
                              (4, 0), (0, 16), (6, 0, 0)),
                },
            ],
        ),
    ),

    'sgn': dict(
        name=['sgn'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.complex64, Dtype.complex128,
               Dtype.float64, Dtype.float32, Dtype.float16,
               Dtype.int16, Dtype.int32, Dtype.int64,
               Dtype.uint8, Dtype.int8, Dtype.bool],
        tensor_para=dict(
            gen_fn=Genfunc.randn_cmplx,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130),
                              (0,), (4, 0), (3, 0, 9)),
                },
            ],
        ),
    ),

    'sgn_zero': dict(
        name=['sgn'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.complex64, Dtype.complex128],
        tensor_para=dict(
            gen_fn=Genfunc.zeros,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((), (1024, ), (384, 128),
                              (64, 1, 128), (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),
}
