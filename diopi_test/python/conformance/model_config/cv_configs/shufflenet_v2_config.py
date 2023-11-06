import numpy as np

shufflenet_v2_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            output_size=[(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 1024, 7, 7), (15, 1024, 7, 7), (64, 1024, 7, 7), (64, 1024, 7, 7)],
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
            alpha=[4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05, 4e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(58, 58, 1, 1), (1000,), (58, 24, 1, 1), (232, 1, 3, 3), (232, 232, 1, 1), (116, 1, 3, 3), (116, 116, 1, 1), (24, 1, 3, 3), (24, 3, 3, 3), (58, 1, 3, 3), (1024, 464, 1, 1), (1000, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(58, 58, 1, 1), (1000,), (58, 24, 1, 1), (232, 1, 3, 3), (232, 232, 1, 1), (116, 1, 3, 3), (116, 116, 1, 1), (24, 1, 3, 3), (24, 3, 3, 3), (58, 1, 3, 3), (1024, 464, 1, 1), (1000, 1024)],
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
            alpha=[-0.0370167, -0.0929414, -0.171371, -0.0092167, -0.148716, -0.0257768],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(58, 24, 1, 1), (1024, 464, 1, 1), (1000,), (58,), (1000, 1024), (1000, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(58, 24, 1, 1), (1024, 464, 1, 1), (1000,), (58,), (1000, 1024), (1000, 1024)],
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
            other=[1],
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
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
            other=[0],
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
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
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 232, 14, 14), (16, 232, 7, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(232,), (232,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(232,), (232,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(232,), (232,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(232,), (232,)],
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
            dim=[1, 1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(64, 232, 7, 7), (64, 232, 7, 7)], [(15, 232, 7, 7), (15, 232, 7, 7)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(648,), (24,), (24,), (216,), (24,), (24,), (1392,), (58,), (58,), (1392,), (58,), (58,), (522,), (58,), (58,), (3364,), (58,), (58,), (3364,), (58,), (58,), (522,), (58,), (58,), (3364,), (58,), (58,), (3364,), (58,), (58,), (522,), (58,), (58,), (3364,), (58,), (58,), (3364,), (58,), (58,), (522,), (58,), (58,), (3364,), (58,), (58,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (13456,), (116,), (116,), (1044,), (116,), (116,), (13456,), (116,), (116,), (2088,), (232,), (232,), (53824,), (232,), (232,), (53824,), (232,), (232,), (2088,), (232,), (232,), (53824,), (232,), (232,), (53824,), (232,), (232,), (2088,), (232,), (232,), (53824,), (232,), (232,), (53824,), (232,), (232,), (2088,), (232,), (232,), (53824,), (232,), (232,), (53824,), (232,), (232,), (2088,), (232,), (232,), (53824,), (232,), (232,), (475136,), (1024,), (1024,), (1024000,), (1000,), (3,), (3,), (24,), (24,), (24,), (24,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (58,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (116,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (232,), (1024,), (1024,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                    "gen_policy": "gen_tensor_list_diff_shape",
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
                    "requires_grad": [False],
                    "shape": [(16, 116, 14, 14), (64, 58, 28, 28)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(116, 116, 1, 1), (58, 58, 1, 1)],
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
                    "shape": [(15, 3, 224, 224), (64, 3, 224, 224), (16, 3, 224, 224)],
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
            other=[1, 64, 15, 64, 15, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), (), (), ()],
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
            value=[1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (64, 116, 14, 14), (64, 58, 28, 28), (232,), (1000,)],
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
                    "shape": [(15, 3, 224, 224), (16, 3, 224, 224), (64, 3, 224, 224)],
                    "dtype": [np.uint8],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(64, 1024), (64, 1024), (15, 1024), (16, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1000, 1024), (1000, 1024), (1000, 1024), (1000, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1000,), (1000,), (1000,), (1000,)],
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
                    "shape": [(15, 1000), (64, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        interface=["torch.nn.functional"],
        requires_backward=[0],
        para=dict(
            kernel_size=[(3, 3), (3, 3), (3, 3), (3, 3)],
            stride=[(2, 2), (2, 2), (2, 2), (2, 2)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            ceil_mode=[False, False, False, False],
            return_indices=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 24, 112, 112), (64, 24, 112, 112), (64, 24, 112, 112), (15, 24, 112, 112)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(232,), (116,), (116, 116, 1, 1), (116, 1, 3, 3), (1024, 464, 1, 1), (58, 24, 1, 1), (1000,), (58, 58, 1, 1), (1000, 1024), (), (24, 3, 3, 3), (232, 232, 1, 1), (24,), (1024,), (58, 1, 3, 3), (58,), (24, 1, 3, 3), (232, 1, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(24, 3, 3, 3), (24,), (116,), (58,), (116, 116, 1, 1), (232,), (24, 1, 3, 3), (1000, 1024), (232, 1, 3, 3), (116, 1, 3, 3), (1024, 464, 1, 1), (58, 1, 3, 3), (58, 24, 1, 1), (1024,), (58, 58, 1, 1), (1000,), (232, 232, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(58, 1, 3, 3), (116, 116, 1, 1), (58, 58, 1, 1), (24,), (), (), (1024,), (232,), (1000, 1024), (116, 1, 3, 3), (1000,), (1024, 464, 1, 1), (58,), (58, 24, 1, 1), (24, 3, 3, 3), (116,), (232, 1, 3, 3), (232, 232, 1, 1), (24, 1, 3, 3)],
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
                    "shape": [(64, 1000), (15, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "requires_grad": [False],
                    "shape": [(64,), (15,)],
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
            mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.00862069, 0.096225, 0.01, 0.01, 1, 1, 0.00431034, 0.00215517, 1, 1, 0.0416667, 0.0172414],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(116, 116, 1, 1), (24, 3, 3, 3), (24, 3, 3, 3), (1000, 1024), (232, 1, 3, 3), (58, 1, 3, 3), (232, 232, 1, 1), (1024, 464, 1, 1), (116, 1, 3, 3), (24, 1, 3, 3), (58, 24, 1, 1), (58, 58, 1, 1)],
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
                    "shape": [(15, 58, 28, 28), (64, 24, 112, 112)],
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
                    "shape": [(16, 1000), (64, 1000)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(16, 3, 224, 224), (15, 3, 224, 224), (64, 3, 224, 224)],
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
            dim=[None, None],
            keepdim=[False, False],
            dtype=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (15,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
