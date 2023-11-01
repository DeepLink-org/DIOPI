from ...config import Genfunc
from ...diopi_runtime import Dtype

inceptionv3_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            output_size=[(1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 2048, 8, 8), (32, 2048, 8, 8), (16, 2048, 8, 8)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (320,), (192, 128, 7, 1), (64, 256, 1, 1), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(128,), (320,), (192, 128, 7, 1), (64, 256, 1, 1), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_2': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[1, 1, 1, 1, 1, -0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 192, 35, 35), (64, 256, 1, 1), (80,), (192,), (1000, 2048), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 192, 35, 35), (64, 256, 1, 1), (80,), (192,), (1000, 2048), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0, 0, 1, 0],
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(48, 288, 1, 1), (64, 32, 3, 3), (), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[0],
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'argmax': dict(
        name=["argmax"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
            keepdim=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            ceil_mode=[False, False, False, False, False, False],
            count_include_pad=[True, True, True, True, True, True],
            divisor_override=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 256, 35, 35), (32, 288, 35, 35), (32, 2048, 8, 8), (32, 768, 17, 17), (32, 1280, 8, 8), (32, 192, 35, 35)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-01,
        rtol=1e-02,
        atol_half=1e-01,
        rtol_half=1e-02,
        interface=["torch.nn.functional"],
        para=dict(
            training=[False, False],
            momentum=[0.1, 0.1],
            eps=[0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 192, 8, 8), (16, 48, 35, 35)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(192,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(192,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(192,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(192,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1, 0, 0, 1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(32, 192, 17, 17), (32, 192, 17, 17), (32, 192, 17, 17), (32, 192, 17, 17)], [(32, 320, 8, 8), (32, 768, 8, 8), (32, 768, 8, 8), (32, 192, 8, 8)], [(16, 384, 8, 8), (16, 384, 8, 8)], [(32, 384, 8, 8), (32, 384, 8, 8)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(16, 320, 8, 8), (16, 192, 8, 8), (16, 768, 8, 8)], [(32, 384, 17, 17), (32, 96, 17, 17), (32, 288, 17, 17)], [(864,), (32,), (32,), (9216,), (32,), (32,), (18432,), (64,), (64,), (5120,), (80,), (80,), (138240,), (192,), (192,), (12288,), (64,), (64,), (9216,), (48,), (48,), (76800,), (64,), (64,), (12288,), (64,), (64,), (55296,), (96,), (96,), (82944,), (96,), (96,), (6144,), (32,), (32,), (16384,), (64,), (64,), (12288,), (48,), (48,), (76800,), (64,), (64,), (16384,), (64,), (64,), (55296,), (96,), (96,), (82944,), (96,), (96,), (16384,), (64,), (64,), (18432,), (64,), (64,), (13824,), (48,), (48,), (76800,), (64,), (64,), (18432,), (64,), (64,), (55296,), (96,), (96,), (82944,), (96,), (96,), (18432,), (64,), (64,), (995328,), (384,), (384,), (18432,), (64,), (64,), (55296,), (96,), (96,), (82944,), (96,), (96,), (147456,), (192,), (192,), (98304,), (128,), (128,), (114688,), (128,), (128,), (172032,), (192,), (192,), (98304,), (128,), (128,), (114688,), (128,), (128,), (114688,), (128,), (128,), (114688,), (128,), (128,), (172032,), (192,), (192,), (147456,), (192,), (192,), (147456,), (192,), (192,), (122880,), (160,), (160,), (179200,), (160,), (160,), (215040,), (192,), (192,), (122880,), (160,), (160,), (179200,), (160,), (160,), (179200,), (160,), (160,), (179200,), (160,), (160,), (215040,), (192,), (192,), (147456,), (192,), (192,), (147456,), (192,), (192,), (122880,), (160,), (160,), (179200,), (160,), (160,), (215040,), (192,), (192,), (122880,), (160,), (160,), (179200,), (160,), (160,), (179200,), (160,), (160,), (179200,), (160,), (160,), (215040,), (192,), (192,), (147456,), (192,), (192,), (147456,), (192,), (192,), (147456,), (192,), (192,), (258048,), (192,), (192,), (258048,), (192,), (192,), (147456,), (192,), (192,), (258048,), (192,), (192,), (258048,), (192,), (192,), (258048,), (192,), (192,), (258048,), (192,), (192,), (147456,), (192,), (192,), (147456,), (192,), (192,), (552960,), (320,), (320,), (147456,), (192,), (192,), (258048,), (192,), (192,), (258048,), (192,), (192,), (331776,), (192,), (192,), (409600,), (320,), (320,), (491520,), (384,), (384,), (442368,), (384,), (384,), (442368,), (384,), (384,), (573440,), (448,), (448,), (1548288,), (384,), (384,), (442368,), (384,), (384,), (442368,), (384,), (384,), (245760,), (192,), (192,), (655360,), (320,), (320,), (786432,), (384,), (384,), (442368,), (384,), (384,), (442368,), (384,), (384,), (917504,), (448,), (448,), (1548288,), (384,), (384,), (442368,), (384,), (384,), (442368,), (384,), (384,), (393216,), (192,), (192,), (2048000,), (1000,), (3,), (3,), (32,), (32,), (32,), (32,), (64,), (64,), (80,), (80,), (192,), (192,), (64,), (64,), (48,), (48,), (64,), (64,), (64,), (64,), (96,), (96,), (96,), (96,), (32,), (32,), (64,), (64,), (48,), (48,), (64,), (64,), (64,), (64,), (96,), (96,), (96,), (96,), (64,), (64,), (64,), (64,), (48,), (48,), (64,), (64,), (64,), (64,), (96,), (96,), (96,), (96,), (64,), (64,), (384,), (384,), (64,), (64,), (96,), (96,), (96,), (96,), (192,), (192,), (128,), (128,), (128,), (128,), (192,), (192,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (192,), (192,), (192,), (192,), (192,), (192,), (160,), (160,), (160,), (160,), (192,), (192,), (160,), (160,), (160,), (160,), (160,), (160,), (160,), (160,), (192,), (192,), (192,), (192,), (192,), (192,), (160,), (160,), (160,), (160,), (192,), (192,), (160,), (160,), (160,), (160,), (160,), (160,), (160,), (160,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (320,), (320,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (320,), (320,), (384,), (384,), (384,), (384,), (384,), (384,), (448,), (448,), (384,), (384,), (384,), (384,), (384,), (384,), (192,), (192,), (320,), (320,), (384,), (384,), (384,), (384,), (384,), (384,), (448,), (448,), (384,), (384,), (384,), (384,), (384,), (384,), (192,), (192,)]],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        atol=1e-04,
        rtol=1e-05,
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[-2, -2, -2],
            max=[2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 192, 1, 1), (64, 256, 1, 1), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        interface=["torch.nn.functional"],
        para=dict(
            stride=[(1, 1), (1, 1)],
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 768, 17, 17), (32, 1280, 8, 8)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(160, 768, 1, 1), (384, 1280, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[False],
                    "shape": [None, None],
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 299, 299), (16, 3, 299, 299)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_case_2': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[1, 32, 1, 32, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            p=[0.5],
            train=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'erf': dict(
        name=["erf"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(192, 128, 7, 1), (192, 160, 1, 7), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (96,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'flip': dict(
        name=["flip"],
        interface=["torch"],
        para=dict(
            dims=[(1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 299, 299), (16, 3, 299, 299)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-03,
        rtol=1e-04,
        atol_half=1e-01,
        rtol_half=1e-02,
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 2048), (32, 2048), (16, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1000, 2048), (1000, 2048), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(1000,), (1000,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        interface=["torch.nn.functional"],
        requires_backward=[0],
        para=dict(
            kernel_size=[(3, 3), (3, 3)],
            stride=[(2, 2), (2, 2)],
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            ceil_mode=[False, False],
            return_indices=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 192, 71, 71), (16, 768, 17, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(160, 768, 1, 1), (320, 1280, 1, 1), (48,), (160,), (), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 2048, 1, 1)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.141421, 0.141421, 0.9, 0.9, 0.141421, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(160, 160, 1, 7), (192, 192, 1, 7), (1000,), (96,), (1000, 2048), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000, 2048), (160, 160, 7, 1), (32, 192, 1, 1), (128,), (32,), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            weight=[None],
            reduction=['none'],
            ignore_index=[-100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(32,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        interface=["torch.nn.functional"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 96, 17, 17), (32, 96, 17, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 299, 299), (16, 3, 299, 299)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[None],
            keepdim=[False],
            dtype=[None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'uniform': dict(
        name=["uniform"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        is_inplace=[True],
        para=dict(
            start=[-1, -1, -1],
            end=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(160, 160, 1, 7), (192, 160, 1, 7), (1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
