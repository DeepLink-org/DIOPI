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
        dtype=[Dtype.float32, Dtype.float16, ],
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
                    "shape": ((2, 64, 32, 32),),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    # "shape": ((8, ), (64, ), None, (16, ),
                    #           (7, ), (15, ), None, (16, )),
                    "shape": ((64, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    # "shape": ((8, ), (64, ), None, (16, ),
                    #           (7, ), (15, ), None, (16, )),
                    "shape": ((64, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    # "shape": ((8, ), (64, ), (96, ), (16, ),
                    #           (7, ), (15, ), (96, ), (16, )),
                    "shape": ((64, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16],
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
                    "shape": ((4, 7, 12, 13), (6, 16, 19, 8), (6, 27, 12, 8), (2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1), (2, 256, 200, 304)),
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

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024,), (2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float16, Dtype.float32,
                              Dtype.int16, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        atol=1e-4,
        rtol=1e-5,
        approximate=['none', 'none',
                     'none', 'none', 'none'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((32,), (16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float16, Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[(6, 12), (6, 8), 3, (2, 1), (2, 2)],
            stride=[(3, 100), (3, 2), 2, (2, 1), (2, 1)],
            padding=[(2, 6), (2, 3), 1, 0, (0, 1)],
            dilation=[(4, 3), (2, 3), 1, (1, 1), 1],
            ceil_mode=[True, True, False, True, False],
            return_indices=[False, False, False, False, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((5, 4, 17, 22),
                              (1, 4, 17, 23),
                              (2, 64, 352, 528),
                              (2, 256, 12, 40),
                              (2, 512, 4, 26)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'pointwise_op': dict(
        name=['abs', 'exp', 'floor', 'sqrt', 'rsqrt', 'ceil', 'atan'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32],
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

    # FIXME erfinv输入int或bool报错
    'pointwise_op_int_without_inplace': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp',
        #       'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        name=['abs', 'exp', 'sqrt', 'rsqrt'],
        interface=['torch'],
        dtype=[Dtype.int16, Dtype.int32],
        tensor_para=dict(
            gen_fn=dict(fn=Genfunc.randint, low=-5, high=5),
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


    'pointwise_op_abs_input': dict(
        name=['log', 'log2', 'log10', 'sqrt', 'rsqrt'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.positive,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3), (2, 31, 512, 6, 40),),
                },
            ],
        ),
    ),

    'pow': dict(
        name=['pow'],
        interface=['torch'],
        is_inplace=False,
        para=dict(
            exponent=[-2, -0.5, 0, 0.6, True, 3, 4., 1.],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38)),
                    "dtype": [Dtype.float16, Dtype.float32],
                    "gen_fn": Genfunc.randn,
                }
            ],
        ),
    ),

    'pow_broadcast': dict(
        name=['pow'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 1, 128), (2, 64, 1, 128),
                              (2, 32, 130, 130),(8, 16, 1)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((4, 16), (384, 128), (64, 16, 128),
                              (5, 2, 32, 1, 130)),
                },
            ],
        ),
    ),

    'pow_broadcast_inplace': dict(
        name=['pow'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float16],
        is_inplace=True,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (2, 1024), (2, 384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (5, 2, 32, 130, 130)),
                },
                {
                    "ins": ['exponent'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (2, 32, 1, 130),(16, 1,)),
                },
            ],
        ),
    ),


    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=False,
        dtype=[ Dtype.float32, Dtype.float16, Dtype.int32, Dtype.int16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1)),
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
                    "shape": ((64, ), (2, 1024), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 1, 128),
                              (2, 32, 130, 130),(8, 16, 1)),
                },
                {
                    "ins": ['other'],
                    "shape": ((4, 16), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (5, 2, 32, 1, 130)),
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
                              (5, 2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (2, 32, 1, 130), (16, 1,)),
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
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                    "dtype": [Dtype.int16, Dtype.int32, 
                              Dtype.int8],
                },
            ],
        ),
    ),

    'div': dict(
        name=['div'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32, Dtype.float16],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3), (2, 64, 16, 128),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (64, 1, 128), (2, 32, 1, 1)),
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
                    "shape": ((64, ), (2, 1024), (2, 1, 128),
                              (128, 64, 3, 3), (2, 64, 1, 128),
                              (2, 32, 130, 130), (8, 16, 1)),
                },
                {
                    "ins": ['other'],
                    "shape": ((4, 16), (1024, ), (384, 128),
                              (1, ), (64, 16, 128),
                              (5, 2, 32, 1, 130)),
                },
            ],
        ),
    ),

    'sub_scalar': dict(
        name=['sub'],
        interface=['torch'],
        tag=['scalar'],
        is_inplace=False,
        dtype=[Dtype.float32],
        para=dict(
            other=[0, -1, 0.028, 2.232, 1, -0.2421, -2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128), (2, 64, 128),
                              (128, 64, 3, 3), (128, 32, 2, 2),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'clamp_max_scalar': dict(
        name=['clamp_max'],
        interface=['torch'],
        is_inplace=False,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            max=[True, 4.13, 1, -1, 1e-12, 10, 0, 1.2, -2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152),
                              (384, 128)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_min_scalar': dict(
        name=['clamp_min'],
        interface=['torch'],
        is_inplace=False,
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            min=[0, 1.2, -1.1, 1, 100, 10, -2, 2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152),
                              (384, 128)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        interface=['torch'],
        is_inplace=False,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152)),
                },
                {
                    "ins": ['min'],
                    "shape": ((182, ), (384, 1), (2, 4, 100, 152), (1,)),
                },
                {
                    "ins": ['max'],
                    "shape": ((384, 128), (1, 1, 2)),
                },
            ],
        ),
    ),

    'clamp_max_tensor': dict(
        name=['clamp_max'],
        interface=['torch'],
        is_inplace=False,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182,), (384, 128),
                              (3, 242991, 2,),
                              (2, 4, 100, 152)),
                },
                {
                    "ins": ['max'],
                    "shape": ((1,), (128, ), (3, 1, 2), (4, 100, 152)),
                },
            ],
        ),
    ),

    'clamp_min_tensor': dict(
        name=['clamp_min'],
        interface=['torch'],
        is_inplace=False,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182,), (384, 128),
                              (3, 242991, 2,),
                              (2, 4, 100, 152)),
                },
                {
                    "ins": ['min'],
                    "shape": ((1,), (128, ), (3, 1, 2), (4, 100, 152), (1,)),
                },
            ],
        ),
    ),

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
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1),
                              (1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726)),
                    "dtype": [Dtype.float32, Dtype.float16],
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
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float16],
                    "gen_fn": Genfunc.randn,
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
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16,
                              Dtype.int32, Dtype.int8],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad": [False],
                    "shape": ((12,), (4, 5), (16, 4, 4), (64, 4, 8, 8)),
                    "dtype": [Dtype.float32, Dtype.float16],
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
                    "requires_grad": [False],
                    "shape": ((5,), (5, 3), (16, 8), (1, 800, 1216), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    "shape": ((10,), (3,), (5,), (2,), (30,),
                              (12,), (7,)),
                    "dtype": [Dtype.int32],
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
                    "shape": ((12,), (10, 10), (16, 12), (1, 800, 1216), (4, 4, 14, 14)),
                    "dtype": [Dtype.int32, Dtype.int16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    "shape": ((20,), (10,), (5,), (100,), (10,),
                              (20,), (7,)),
                    "dtype": [Dtype.int32, Dtype.int32, Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=10)
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((13,), (78, 24), (2, 92, 29), (2, 150, 512, 512)),
                    "dtype": [Dtype.float16, Dtype.float32],
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
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((16,), (2, 24), (2, 128, 24), (8, 16, 49, 49), (4, 12, 577, 577)),
                    "dtype": [Dtype.float16, Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'tril': dict(
        name=["tril"],
        interface=["torch"],
        para=dict(
            diagonal=[12],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, 9), (6, 7), (6, 6), (9, 9),
                              (6, 8, 8), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    # FIXME stack输入size为0的张量报错
    'join': dict(
        name=['cat'],
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
                    "shape": ((3, ), (512, 4), (2, 31, 512),
                              (2, 512, 8, 8)),
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16,
                              Dtype.int8, Dtype.int32],
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
                    "dtype": [Dtype.float32, Dtype.float16, Dtype.int16,
                              Dtype.int8, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
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
        dtype=[Dtype.float16, Dtype.float32, Dtype.int16,
               Dtype.int32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((11400, ), (12, 8), (8, 12, 9),
                              (4, 4, 16, 20), (4, 4, 16, 2, 20), (24180,)),
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
            dtype=[Dtype.float16, Dtype.float32, Dtype.int16,
                   Dtype.int32],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 1536, 950),
                              (16, 8), (660, 6, 49, 32)),
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
            negative_slope=[0.01, 0.1, 1, 0.0]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((128,), (16, 7), (64, 28, 28),
                              (2, 32, 208, 304), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float16, Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": ((1,), (182,), (64, 128), (2, 1, 640, 640)),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
            ],
        ),
    ),

    'bitwise_not_int': dict(
        name=['bitwise_not'],
        interface=['torch'],
        is_inplace=False,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((10,), (100, 4), (2, 256, 256)),
                    "dtype": [Dtype.int16, Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, low=-128, high=128),
                },
            ],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        interface=["torch"],
        para=dict(
            dim=[0, -1, 0, 1, 1],
            keepdim=[True, False, True, False, False, True, True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (1024, 80), (2, 256, 256), (2, 1, 64, 64)),
                    "dtype": [ Dtype.float16, Dtype.float32, Dtype.int32, Dtype.int16,
                              Dtype.int8],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'expand': dict(
        name=['expand'],
        interface=['torch.Tensor'],
        para=dict(
            size=[(5,), (8, 2), (4, -1), (60800, 3), (-1, 4), (-1, 8, -1), (7, 3, -1), (5, -1, 8, 6, -1), (4, -1, -1), (-1, -1, 9)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(8,), (60800, 1), (100, 1), (70, 1, 2), (3, 1), (4, 1, 6, 8)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float16,
                              Dtype.int32, Dtype.int16],
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
                    "requires_grad": [False],
                    "shape": ((8,), (3, 9), (16, 4, 4), (64, 4, 14, 14), (64, 4, 16, 16)),
                    "dtype": [Dtype.float32, Dtype.float16],
                },
                {
                    "ins": ['index'],
                    "shape": ((12,), (2, 15), (16, 4, 4), (64, 4, 14, 14), (64, 4, 16, 16)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ],
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

    'layer_norm': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32, Dtype.float16],
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
                    "shape": ((128,), (64,), (32,),
                              (3, 5), (2, 16, 128)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": ((128,), (64,), (32,),
                              (3, 5), (2, 16, 128)),
                },
            ]
        )
    ),

    # FIXME interpolate输入mode为linear，做down sample精度不一致
    'interpolate': dict(
        name=["interpolate"],
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            mode=['nearest', 'nearest', 'nearest', 'nearest', 'nearest',
                  'bilinear', 'bilinear'],
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
                    "shape": ((2, 256, 25, 38), (2, 2, 16, 16), (2, 2, 16, 16),
                              (2, 256, 13, 19), (3, 12, 14, 19), (2, 16, 1, 1), (2, 16, 15, 32)),
                },
            ]
        )
    ),

    'flip': dict(
        name=['flip'],
        interface=['torch'],
        para=dict(
            dims=[(-1,), (0,), (1,), 
                  (-1,), (0, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((12,), (49, 49)),
                    "dtype": [Dtype.float16, Dtype.float32,
                              Dtype.int16, Dtype.int32],
                    "gen_fn": Genfunc.randn,
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
                    "dtype": [Dtype.float32],
                },
                {
                    "ins": ['out'],
                    "shape": [(32, 64,), (128, 24, 32), (16, 8,), (24, 12,)],
                    "dtype": [Dtype.float16],
                },
            ]
        ),
    ),

}
