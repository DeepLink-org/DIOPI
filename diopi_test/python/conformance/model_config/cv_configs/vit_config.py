import numpy as np

vit_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 577, 768), (15, 577, 768), (16, 577, 768), (16, 577, 768), (64, 577, 768), (64, 577, 768), (32, 577, 768), (32, 577, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 577, 768), (1, 577, 768), (16, 577, 768), (1, 577, 768), (1, 577, 768), (64, 577, 768), (32, 577, 768), (1, 577, 768)],
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
            alpha=[0.1, 0.1, 0.1, 1, 1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2304,), (768,), (1, 1, 768), (3, 32, 12, 577, 64), (32, 577, 768), (1000,), (3072,), (1000, 768), (15, 577, 768), (3072, 768), (768, 3072), (2304, 768), (1, 577, 768), (768, 768), (3, 15, 12, 577, 64), (768, 3, 16, 16)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(2304,), (768,), (1, 1, 768), (3, 32, 12, 577, 64), (32, 577, 768), (1000,), (3072,), (1000, 768), (15, 577, 768), (3072, 768), (768, 3072), (2304, 768), (1, 577, 768), (768, 768), (3, 15, 12, 577, 64), (768, 3, 16, 16)],
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
            other=[9.09091e-05, 0, 9.09091e-05],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 1000), (1, 577, 768), (32, 1000)],
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
            other=[1e-08, 1e-08, 1e-08, 1e-08, 0, 1e-06, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3072,), (768, 3072), (768,), (1000, 768), (), (), (768, 768), (2304, 768), (2304,), (768, 3, 16, 16), (1, 577, 768), (3072, 768), (1, 1, 768), (1000,)],
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
            value=[-9.76578e-05, -5.63676e-05, -4.23594e-05, -8.60746e-05, -7.21814e-05, -1.48151e-05, -1.01799e-05, -9.22233e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2304,), (768,), (1, 1, 768), (1, 1, 768), (768, 768), (768, 768), (768, 3, 16, 16), (768, 3, 16, 16)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(2304,), (768,), (1, 1, 768), (1, 1, 768), (768, 768), (768, 768), (768, 3, 16, 16), (768, 3, 16, 16)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(2304,), (768,), (1, 1, 768), (1, 1, 768), (768, 768), (768, 768), (768, 3, 16, 16), (768, 3, 16, 16)],
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
            value=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (1, 1, 768), (1000,), (768,), (2304,), (2304, 768), (768, 3072), (1, 577, 768), (1000, 768), (768, 3, 16, 16), (3072, 768), (3072,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (1, 1, 768), (1000,), (768,), (2304,), (2304, 768), (768, 3072), (1, 577, 768), (1000, 768), (768, 3, 16, 16), (3072, 768), (3072,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (1, 1, 768), (1000,), (768,), (2304,), (2304, 768), (768, 3072), (1, 577, 768), (1000, 768), (768, 3, 16, 16), (3072, 768), (3072,)],
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

    'bmm': dict(
        name=["bmm"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(768, 577, 64), (384, 577, 577), (384, 577, 577), (384, 577, 577), (192, 577, 577), (180, 577, 64), (180, 577, 64), (180, 577, 577), (180, 577, 577), (180, 577, 577), (768, 577, 577), (192, 577, 64), (384, 577, 64), (384, 577, 64), (384, 64, 577), (180, 64, 577)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mat2"],
                    "requires_grad": [True],
                    "shape": [(768, 64, 577), (384, 577, 64), (384, 577, 64), (384, 577, 64), (192, 577, 64), (180, 64, 577), (180, 64, 577), (180, 577, 64), (180, 577, 64), (180, 577, 64), (768, 577, 64), (192, 64, 577), (384, 64, 577), (384, 64, 577), (384, 577, 577), (180, 577, 577)],
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
            dim=[1, 1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(32, 1, 768), (32, 576, 768)], [(15, 1, 768), (15, 576, 768)], [(2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (768000,), (1000,), (3,), (3,)], [(768,), (443136,), (589824,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (1769472,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
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
            min=[-2],
            max=[2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 577, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'clamp_case_2': dict(
        name=["clamp"],
        atol=1e-04,
        rtol=1e-05,
        interface=["torch"],
        para=dict(
            min=[None],
            max=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [()],
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
            stride=[(16, 16), (16, 16), (16, 16), (16, 16)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 3, 384, 384), (32, 3, 384, 384), (64, 3, 384, 384), (15, 3, 384, 384)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(768, 3, 16, 16), (768, 3, 16, 16), (768, 3, 16, 16), (768, 3, 16, 16)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(768,), (768,), (768,), (768,)],
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
                    "shape": [(32, 3, 384, 384), (15, 3, 384, 384), (16, 3, 384, 384), (64, 3, 384, 384)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
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
            other=[0.988996, 0.993668, 0.921286, 0.996095, 0.954746, 0.98226, 0.997942, 0.998479],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(768, 768), (1000, 768), (3072,), (3072,), (1, 577, 768), (1, 1, 768), (768, 3, 16, 16), (768, 3, 16, 16)],
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
            other=[15, 32, 15, 1, 1, 32, 1, 8, 8, 8, 8, 8, 8],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), (), (), (), (15, 12, 577, 577), (15, 12, 577, 577), (64, 12, 577, 577), (32, 12, 577, 577), (32, 12, 577, 577), (16, 12, 577, 577)],
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
            p=[0.1, 0.1, 0.1, 0.1],
            training=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 577, 768), (32, 577, 3072), (15, 577, 768), (15, 577, 3072)],
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
                    "shape": [(1, 577, 768)],
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
            value=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 1, 768), (1, 577, 768), (3, 32, 12, 577, 64), (3, 15, 12, 577, 64), (32, 1000), (768, 3072), (768,), (2304,), (), (768, 3, 16, 16)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'flip': dict(
        name=["flip"],
        interface=["torch"],
        para=dict(
            dims=[(1,), (1,), (1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 384, 384), (15, 3, 384, 384), (16, 3, 384, 384), (64, 3, 384, 384)],
                    "dtype": [np.uint8],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'gelu': dict(
        name=["gelu"],
        interface=["torch.nn.functional"],
        para=dict(
            approximate=['none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 577, 3072), (64, 577, 3072), (15, 577, 3072), (32, 577, 3072)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        interface=["torch.nn.functional"],
        para=dict(
            normalized_shape=[(768,), (768,), (768,), (768,)],
            eps=[1e-06, 1e-06, 1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 577, 768), (15, 577, 768), (32, 577, 768), (64, 577, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(768,), (768,), (768,), (768,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(768,), (768,), (768,), (768,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_mean"],
                    "requires_grad": [False],
                    "shape": [(16, 577, 1), (15, 577, 1), (32, 577, 1), (64, 577, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_invstd"],
                    "requires_grad": [False],
                    "shape": [(16, 577, 1), (15, 577, 1), (32, 577, 1), (64, 577, 1)],
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
                    "shape": [(64, 577, 768), (32, 577, 3072), (15, 768), (16, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(3072, 768), (768, 3072), (1000, 768), (1000, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(3072,), (768,), (1000,), (1000,)],
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
                    "shape": [(32, 1000), (15, 1000)],
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
                    "shape": [(32,), (32, 1), (15,), (15, 1)],
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
            dim=[None, None, None, None, None, None, None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(768, 3, 16, 16), (1000,), (768,), (2304,), (768, 768), (3072,), (2304, 768), (768, 3072), (), (1000, 768), (1, 1, 768), (3072, 768), (1, 577, 768)],
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
                    "shape": [(32,), (15,)],
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
                    "shape": [(15, 1000), (15, 1000), (32, 1000), (32, 1000), (15, 577, 768), (32, 577, 3072), (15, 577, 3072), (32, 577, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 1000), (15, 1000), (32, 1000), (32, 1000), (15, 577, 768), (32, 577, 3072), (15, 577, 3072), (32, 577, 768)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(768, 768), (2304, 768), (3072, 768), (1, 577, 768), (768,), (2304,), (3072,), (768, 3, 16, 16), (1, 1, 768), (1000, 768), (768, 3072), (1000,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (), (), (), (), (), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.999975, 0.999999, 0.999988, 0.999976, 0.999987, 0.999997, 1, 0.999],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(768, 768), (1000, 768), (2304,), (3072,), (768, 3, 16, 16), (768, 3, 16, 16), (1, 577, 768), (1, 1, 768)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul_case_4': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1.11111, 1.11111, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 577, 3072), (15, 577, 768), (1000, 768), (2304, 768), (1000,), (768,), (768, 3, 16, 16), (), ()],
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
                    "shape": [(15, 1000), (32, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            dim=[(0,), (0, 1), (0, 1), (0, 1, 2), (0, 1), (0, 1, 2, 3), (0,), (0,), (0, 1, 2), (0,), (0, 1), (0, 1), (0,)],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1000,), (1000, 768), (2304, 768), (1, 577, 768), (3072, 768), (768, 3, 16, 16), (2304,), (152,), (1, 1, 768), (3072,), (768, 768), (768, 3072), (768,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            mean=[0, 0, 0],
            std=[0.0360844, 1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(768, 3, 16, 16), (768,), (3072,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [()],
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
            dim=[-1, -1],
            value=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(15, 1000), (32, 1000)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["index"],
                    "requires_grad": [False],
                    "shape": [(15, 1), (32, 1)],
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
            dim=[-1, -1, 1, -1, -1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 12, 577, 577), (64, 12, 577, 577), (16, 1000), (32, 12, 577, 577), (15, 12, 577, 577), (64, 1000)],
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
                    "shape": [(2304,), (3072,), (2304, 768), (768, 3072), (3072, 768), (768, 3, 16, 16), (1000, 768), (768,), (1, 577, 768), (768, 768), (1, 1, 768), (1000,)],
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
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 384, 384), (15, 3, 384, 384), (16, 3, 384, 384), (32, 3, 384, 384)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
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
            dim=[[0], [0], None, [-1], [0], [0], None, [-1]],
            keepdim=[True, True, False, False, True, True, False, False],
            dtype=[None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 577, 768), (32, 1, 768), (15,), (15, 1000), (15, 1, 768), (15, 577, 768), (32,), (32, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
            start=[-1, -0.0395285, -0.0395285],
            end=[1, 0.0395285, 0.0395285],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 577, 768), (3072, 768), (768, 3072)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
