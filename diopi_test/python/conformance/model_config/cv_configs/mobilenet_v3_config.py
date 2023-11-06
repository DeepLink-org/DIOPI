import numpy as np

mobilenet_v3_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            output_size=[(1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(15, 960, 7, 7), (80, 120, 28, 28)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1e-05, 1, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(120, 40, 1, 1), (128, 24, 56, 56), (24,), (240,), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(120, 40, 1, 1), (128, 24, 56, 56), (24,), (240,), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add_case_2': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[-0.064, -0.064, 1, 1, -0.064, -0.064],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(112,), (1280,), (15, 480, 14, 14), (128, 40, 28, 28), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(112,), (1280,), (15, 480, 14, 14), (128, 40, 28, 28), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.0316, 0.0316, 0.0316, 0.0316, 0.0316, 0.0316, 1],
            alpha=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 16, 1, 1), (200, 80, 1, 1), (480,), (120,), (1000, 1280), (1280, 960), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[3, 3, 0],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 120, 1, 1), (128, 672, 1, 1), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(24, 72, 1, 1), (120, 40, 1, 1), (672,), (160,), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(24, 72, 1, 1), (120, 40, 1, 1), (672,), (160,), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(24, 72, 1, 1), (120, 40, 1, 1), (672,), (160,), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'addcmul': dict(
        name=["addcmul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1280,), (1000,), (1280, 960), (1000, 1280), (32, 120, 1, 1), (960, 160, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(1280,), (1000,), (1280, 960), (1000, 1280), (32, 120, 1, 1), (960, 160, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(1280,), (1000,), (1280, 960), (1000, 1280), (32, 120, 1, 1), (960, 160, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "requires_grad": [False],
                    "shape": [(128, 1000), (80, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
            training=[True, True],
            momentum=[0.01, 0.01],
            eps=[0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(128, 240, 28, 28), (15, 112, 14, 14)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(240,), (112,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(240,), (112,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(240,), (112,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(240,), (112,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(432,), (16,), (16,), (144,), (16,), (16,), (256,), (16,), (16,), (1024,), (64,), (64,), (576,), (64,), (64,), (1536,), (24,), (24,), (1728,), (72,), (72,), (648,), (72,), (72,), (1728,), (24,), (24,), (1728,), (72,), (72,), (1800,), (72,), (72,), (1728,), (24,), (1728,), (72,), (2880,), (40,), (40,), (4800,), (120,), (120,), (3000,), (120,), (120,), (3840,), (32,), (3840,), (120,), (4800,), (40,), (40,), (4800,), (120,), (120,), (3000,), (120,), (120,), (3840,), (32,), (3840,), (120,), (4800,), (40,), (40,), (9600,), (240,), (240,), (2160,), (240,), (240,), (19200,), (80,), (80,), (16000,), (200,), (200,), (1800,), (200,), (200,), (16000,), (80,), (80,), (14720,), (184,), (184,), (1656,), (184,), (184,), (14720,), (80,), (80,), (14720,), (184,), (184,), (1656,), (184,), (184,), (14720,), (80,), (80,), (38400,), (480,), (480,), (4320,), (480,), (480,), (57600,), (120,), (57600,), (480,), (53760,), (112,), (112,), (75264,), (672,), (672,), (6048,), (672,), (672,), (112896,), (168,), (112896,), (672,), (75264,), (112,), (112,), (75264,), (672,), (672,), (16800,), (672,), (672,), (112896,), (168,), (112896,), (672,), (107520,), (160,), (160,), (153600,), (960,), (960,), (24000,), (960,), (960,), (230400,), (240,), (230400,), (960,), (153600,), (160,), (160,), (153600,), (960,), (960,), (24000,), (960,), (960,), (230400,), (240,), (230400,), (960,), (153600,), (160,), (160,), (153600,), (960,), (960,), (1228800,), (1280,), (1280000,), (1000,), (3,), (3,), (16,), (16,), (16,), (16,), (16,), (16,), (64,), (64,), (64,), (64,), (24,), (24,), (72,), (72,), (72,), (72,), (24,), (24,), (72,), (72,), (72,), (72,), (40,), (40,), (120,), (120,), (120,), (120,), (40,), (40,), (120,), (120,), (120,), (120,), (40,), (40,), (240,), (240,), (240,), (240,), (80,), (80,), (200,), (200,), (200,), (200,), (80,), (80,), (184,), (184,), (184,), (184,), (80,), (80,), (184,), (184,), (184,), (184,), (80,), (80,), (480,), (480,), (480,), (480,), (112,), (112,), (672,), (672,), (672,), (672,), (112,), (112,), (672,), (672,), (672,), (672,), (160,), (160,), (960,), (960,), (960,), (960,), (160,), (160,), (960,), (960,), (960,), (960,), (160,), (160,), (960,), (960,)]],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        atol=1e-04,
        rtol=1e-05,
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[0, 0],
            max=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(80, 72, 1, 1), (128, 960, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
            stride=[(2, 2), (1, 1)],
            padding=[(2, 2), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[72, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(128, 72, 56, 56), (128, 672, 7, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(72, 1, 5, 5), (160, 672, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [False],
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
                    "shape": [(80, 3, 224, 224), (128, 3, 224, 224), (15, 3, 224, 224)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'div_case_2': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[6, 6, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 120, 1, 1), (15, 672, 1, 1), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            p=[0.2, 0.2],
            training=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(15, 1280), (128, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 1, 0, 0, 0, 1, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(184,), (184,), (24, 64, 1, 1), (16, 1, 3, 3), (), (), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'flip': dict(
        name=["flip"],
        interface=["torch"],
        para=dict(
            dims=[(1,), (1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 3, 224, 224), (128, 3, 224, 224), (80, 3, 224, 224)],
                    "dtype": [np.uint8],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 72, 1, 1), (15, 120, 1, 1), (128, 672, 1, 1), (15, 72, 1, 1), (128, 120, 1, 1), (128, 960, 1, 1), (15, 960, 1, 1), (128, 480, 1, 1), (15, 672, 1, 1), (15, 480, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'hardswish': dict(
        name=["hardswish"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(80, 1280), (128, 1280), (128, 1280), (15, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'hardswish_case_2': dict(
        name=["hardswish"],
        interface=["torch.nn.functional"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(128, 200, 14, 14), (128, 960, 7, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'le': dict(
        name=["le"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 960, 1, 1), (128, 480, 1, 1), (128, 72, 1, 1), (128, 120, 1, 1), (15, 72, 1, 1), (15, 960, 1, 1), (15, 672, 1, 1), (128, 672, 1, 1), (15, 480, 1, 1), (15, 120, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "requires_grad": [True],
                    "shape": [(128, 960), (128, 960), (128, 1280), (128, 1280), (15, 960), (15, 1280), (80, 960), (80, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1280, 960), (1280, 960), (1000, 1280), (1000, 1280), (1280, 960), (1000, 1280), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1280,), (1280,), (1000,), (1000,), (1280,), (1000,), (1280,), (1000,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(128, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'logical_and': dict(
        name=["logical_and"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 480, 1, 1), (128, 960, 1, 1), (15, 672, 1, 1), (15, 120, 1, 1), (128, 72, 1, 1), (128, 120, 1, 1), (15, 480, 1, 1), (128, 672, 1, 1), (15, 72, 1, 1), (15, 960, 1, 1)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "shape": [(128, 480, 1, 1), (128, 960, 1, 1), (15, 672, 1, 1), (15, 120, 1, 1), (128, 72, 1, 1), (128, 120, 1, 1), (15, 480, 1, 1), (128, 672, 1, 1), (15, 72, 1, 1), (15, 960, 1, 1)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(112,), (32,), (184, 1, 3, 3), (240, 1, 3, 3), (), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(80, 672, 14, 14), (80, 480, 14, 14), (15, 1280), (128, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(80, 672, 1, 1), (80, 480, 1, 1), (15, 1280), (128, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(72, 1, 5, 5), (960, 160, 1, 1), (40,), (160,), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1.25, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(480, 1, 3, 3), (184, 1, 3, 3), (1000,), (184,), (), (), (128, 1280), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            weight=[None, None],
            reduction=['none', 'none'],
            ignore_index=[-100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(128, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "requires_grad": [False],
                    "shape": [(128,), (15,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            mean=[0, 0, 0, 0],
            std=[0.104257, 0.176777, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(184, 80, 1, 1), (64, 16, 1, 1), (1000, 1280), (1280, 960)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(15, 64, 56, 56), (15, 120, 28, 28)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "requires_grad": [False],
                    "shape": [(80, 1000), (128, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sqrt': dict(
        name=["sqrt"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(120, 32, 1, 1), (200, 80, 1, 1), (480,), (72,), (1280, 960), (1000, 1280)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 3, 224, 224), (15, 3, 224, 224), (80, 3, 224, 224)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], None, [2, 3], [2, 3], [2, 3], None, [2, 3], [2, 3], [2, 3], [2, 3]],
            keepdim=[True, True, True, True, True, False, True, True, True, False, True, True, True, True],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 120, 28, 28), (15, 672, 7, 7), (128, 672, 14, 14), (128, 72, 28, 28), (128, 480, 14, 14), (15,), (15, 960, 7, 7), (15, 480, 14, 14), (15, 672, 14, 14), (128,), (15, 120, 28, 28), (15, 72, 28, 28), (128, 672, 7, 7), (128, 960, 7, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'where': dict(
        name=["where"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["condition"],
                    "requires_grad": [False],
                    "shape": [(128, 960, 1, 1), (15, 120, 1, 1), (15, 480, 1, 1), (128, 72, 1, 1), (15, 672, 1, 1), (15, 72, 1, 1), (128, 672, 1, 1), (128, 480, 1, 1), (15, 960, 1, 1), (128, 120, 1, 1)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(128, 960, 1, 1), (15, 120, 1, 1), (15, 480, 1, 1), (128, 72, 1, 1), (15, 672, 1, 1), (15, 72, 1, 1), (128, 672, 1, 1), (128, 480, 1, 1), (15, 960, 1, 1), (128, 120, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(), (), (), (), (), (), (), (), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
