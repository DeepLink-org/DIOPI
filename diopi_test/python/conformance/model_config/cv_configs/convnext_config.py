import numpy as np

convnext_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 96, 56, 56), (64, 192, 28, 28), (64, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 96, 56, 56), (64, 192, 28, 28), (64, 1000), (15, 1000)],
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
            alpha=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1536, 384), (192, 768), (384, 192, 2, 2), (192, 96, 2, 2), (192,), (96,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1536, 384), (192, 768), (384, 192, 2, 2), (192, 96, 2, 2), (192,), (96,)],
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
            other=[0, 0, 0, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(96, 3, 4, 4), (192, 768), (1536, 384), (64, 1000), (96, 1, 7, 7), (768, 384, 2, 2), (384, 96), (3072, 768), (768, 3072), (96, 384), (192, 1, 7, 7), (384, 1, 7, 7), (768, 1, 7, 7), (1000, 768), (768, 192), (384, 192, 2, 2), (192, 96, 2, 2), (384, 1536), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0],
            alpha=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 192, 2, 2), (192, 96, 2, 2), (192,), (768,), (768, 192), (1536, 384), ()],
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
            value=[-0.000165924, -9.59308e-05, -0.000130613, -6.15478e-05, -0.000178979, -0.000123128],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(768, 1, 7, 7), (384, 192, 2, 2), (3072, 768), (192, 768), (1536,), (192,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(768, 1, 7, 7), (384, 192, 2, 2), (3072, 768), (192, 768), (1536,), (192,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(768, 1, 7, 7), (384, 192, 2, 2), (3072, 768), (192, 768), (1536,), (192,)],
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
            value=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1000,), (192,), (768, 1, 7, 7), (96, 3, 4, 4), (1536, 384), (192, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(1000,), (192,), (768, 1, 7, 7), (96, 3, 4, 4), (1536, 384), (192, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(1000,), (192,), (768, 1, 7, 7), (96, 3, 4, 4), (1536, 384), (192, 768)],
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
                    "shape": [(64, 1000), (16, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(4608,), (96,), (96,), (96,), (96,), (96,), (73728,), (192,), (192,), (192,), (294912,), (384,), (384,), (384,), (1179648,), (768,), (96,), (4704,), (96,), (96,), (96,), (36864,), (384,), (36864,), (96,), (96,), (4704,), (96,), (96,), (96,), (36864,), (384,), (36864,), (96,), (96,), (4704,), (96,), (96,), (96,), (36864,), (384,), (36864,), (96,), (192,), (9408,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (192,), (9408,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (192,), (9408,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (18816,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (768,), (37632,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (37632,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (37632,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (768000,), (1000,), (3,), (3,)]],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
            min=[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            max=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(768, 3072), (192, 768), (768, 384, 2, 2), (192, 1, 7, 7), (768, 1, 7, 7), (96, 384), (3072, 768), (384, 1, 7, 7), (384, 96), (96, 3, 4, 4), (768, 192), (384, 1536), (384, 192, 2, 2), (1536, 384), (1000, 768), (96, 1, 7, 7), (192, 96, 2, 2)],
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
            stride=[(1, 1), (2, 2)],
            padding=[(3, 3), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[384, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 384, 14, 14), (16, 384, 14, 14)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(384, 1, 7, 7), (768, 384, 2, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(384,), (768,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(64, 3, 224, 224), (16, 3, 224, 224), (15, 3, 224, 224)],
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
        is_inplace=[True],
        para=dict(
            other=[0.982349, 0.999466, 0.998982, 0.99838, 0.998081, 0.986115],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(192, 768), (384, 1536), (96,), (1536,), (96, 3, 4, 4), (192, 96, 2, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'div_case_3': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[0.68, 0.942857, 1, 15],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 384, 14, 14), (15, 192, 28, 28), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "requires_grad": [True],
                    "shape": [(3072, 768), (768, 3072), (768, 1, 7, 7), (96, 384), (192, 1, 7, 7), (384, 1536), (1536, 384), (192, 768), (384, 96), (1000, 768), (768, 384, 2, 2), (768, 192), (384, 1, 7, 7), (96, 3, 4, 4), (192, 96, 2, 2), (384, 192, 2, 2), (96, 1, 7, 7)],
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
            value=[0, 0, 0, 0, 0, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(192, 768), (768, 3072), (1536,), (384,), (768, 1, 7, 7), (384, 192, 2, 2), ()],
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
                    "shape": [(64, 3, 224, 224), (15, 3, 224, 224), (16, 3, 224, 224)],
                    "dtype": [np.uint8],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'floor': dict(
        name=["floor"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64, 1, 1, 1), (15, 1, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'gelu': dict(
        name=["gelu"],
        interface=["torch.nn.functional"],
        para=dict(
            approximate=['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 28, 28, 768), (64, 28, 28, 768), (64, 7, 7, 3072), (64, 7, 7, 3072), (64, 56, 56, 384), (64, 56, 56, 384), (64, 14, 14, 1536), (64, 14, 14, 1536), (15, 14, 14, 1536), (15, 7, 7, 3072), (15, 56, 56, 384), (16, 7, 7, 3072), (16, 28, 28, 768), (16, 56, 56, 384), (16, 14, 14, 1536), (15, 28, 28, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'index': dict(
        name=["index"],
        interface=["CustomizedTest"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64, 3, 104, 86), (64, 3, 184, 201), (64, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(64,), (64,), (64,), (15,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-15, high=15),
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        interface=["torch.nn.functional"],
        para=dict(
            normalized_shape=[(192,), (96,)],
            eps=[1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 28, 28, 192), (16, 56, 56, 96)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(192,), (96,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(192,), (96,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_mean"],
                    "requires_grad": [False],
                    "shape": [(16, 28, 28, 1), (16, 56, 56, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_invstd"],
                    "requires_grad": [False],
                    "shape": [(16, 28, 28, 1), (16, 56, 56, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'lerp': dict(
        name=["lerp"],
        interface=["torch"],
        para=dict(
            weight=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(384, 1, 7, 7), (96, 3, 4, 4), (384,), (768,), (384, 96), (3072, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["end"],
                    "requires_grad": [False],
                    "shape": [(384, 1, 7, 7), (96, 3, 4, 4), (384,), (768,), (384, 96), (3072, 768)],
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
                    "shape": [(15, 7, 7, 768), (64, 7, 7, 768), (16, 768), (15, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(3072, 768), (3072, 768), (1000, 768), (1000, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(3072,), (3072,), (1000,), (1000,)],
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
            dim=[-1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64,), (15,), (16,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None, [-2, -1], None, None],
            keepdim=[False, False, False, False, True, False, False],
            dtype=[None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3072, 768), (1536, 384), (384,), (96,), (64, 768, 7, 7), (768, 1, 7, 7), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'min': dict(
        name=["min"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64,), (15,), (16,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(16, 96, 56, 56), (64, 384, 14, 14), (15, 1000), (64, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 96, 1, 1), (64, 1, 1, 1), (15, 1000), (64, 1000)],
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
            other=[0.999, 0.999, 0.999993, 0.999992, 0.999992, 0.99999],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000,), (96,), (768, 1, 7, 7), (96, 1, 7, 7), (384, 96), (96, 384)],
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
            other=[0.968506, 1, 0.968506, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 1000), (384, 1536), (15, 3, 224, 224), (96, 1, 7, 7), (96,), (1536,), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'neg': dict(
        name=["neg"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'scatter': dict(
        name=["scatter"],
        interface=["torch"],
        para=dict(
            dim=[-1, -1, -1],
            value=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64, 1000), (16, 1000), (15, 1000)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["index"],
                    "requires_grad": [False],
                    "shape": [(64, 1), (16, 1), (15, 1)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(16, 1000), (64, 1000)],
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
                    "shape": [(768, 3072), (1000, 768), (384, 1, 7, 7), (192, 1, 7, 7), (1536,), (96,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,)], [(1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,)], [(1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,)], [(1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,)], [(1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,), (1000,)]],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                    "gen_policy": "gen_tensor_list_diff_shape",
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
                    "shape": [(64, 3, 224, 224), (16, 3, 224, 224), (15, 3, 224, 224)],
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
            dim=[None, [0, 2, 3], [0, 2, 3], [-1], [0, 2, 3], [0, 2, 3], [0], [0, 2, 3], [-1], [0, 2, 3], None, [0, 2, 3], [0, 2, 3]],
            keepdim=[False, True, True, False, True, True, False, True, False, True, False, True, True],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (64, 384, 14, 14), (64, 192, 28, 28), (64, 1000), (64, 768, 7, 7), (64, 96, 56, 56), (1, 1000), (15, 96, 56, 56), (15, 1000), (15, 192, 28, 28), (15,), (15, 768, 7, 7), (15, 384, 14, 14)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
            start=[-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            end=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 192, 2, 2), (768, 3072), (64, 1, 1, 1), (192, 768), (1000, 768), (768, 1, 7, 7), (3072, 768), (384, 1, 7, 7), (384, 1536), (96, 3, 4, 4), (192, 96, 2, 2), (96, 384), (768, 192), (96, 1, 7, 7), (192, 1, 7, 7), (384, 96), (1536, 384), (768, 384, 2, 2), (15, 1, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
