import numpy as np

pointpillars_config = {
    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(864, 7), (912, 7), (7,), (16,)],
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
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7732,), (7544,), (5229, 3), (6582, 3), ()],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["other"],
                    "shape": [(7732,), (7544,), (3,), (3,), ()],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'add_case_2': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[0.0500827, 0.0500225, 0.0500349, 0.0500794, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(42, 384, 1, 1), (128, 64, 3, 3), (12,), (128,), (33751, 64), (35761, 64)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(42, 384, 1, 1), (128, 64, 3, 3), (12,), (128,), (33751, 64), (35761, 64)],
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
            other=[3, 1],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(107136,), ()],
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
            other=[-1, -1, 1, 0, 1e-08, 1e-08, 1e-06, 0],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(38117, 1), (39568, 1), (82,), (23,), (64, 64, 3, 3), (256, 128, 4, 4), (), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[-0.00100228, -0.00100878, -0.00102022, -0.0010207, -0.00100268, -0.00100996],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(128, 128, 2, 2), (256, 256, 3, 3), (18,), (42,), (64, 10), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(128, 128, 2, 2), (256, 256, 3, 3), (18,), (42,), (64, 10), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(128, 128, 2, 2), (256, 256, 3, 3), (18,), (42,), (64, 10), (64, 10)],
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
            value=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(128,), (256, 128, 4, 4), (64, 64, 3, 3), (42,), (256, 256, 3, 3), (128, 128, 2, 2), (256, 128, 3, 3), (12,), (12, 384, 1, 1), (128, 64, 3, 3), (64, 128, 1, 1), (18,), (42, 384, 1, 1), (64,), (128, 128, 3, 3), (18, 384, 1, 1), (256,), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(128,), (256, 128, 4, 4), (64, 64, 3, 3), (42,), (256, 256, 3, 3), (128, 128, 2, 2), (256, 128, 3, 3), (12,), (12, 384, 1, 1), (128, 64, 3, 3), (64, 128, 1, 1), (18,), (42, 384, 1, 1), (64,), (128, 128, 3, 3), (18, 384, 1, 1), (256,), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(128,), (256, 128, 4, 4), (64, 64, 3, 3), (42,), (256, 256, 3, 3), (128, 128, 2, 2), (256, 128, 3, 3), (12,), (12, 384, 1, 1), (128, 64, 3, 3), (64, 128, 1, 1), (18,), (42, 384, 1, 1), (64,), (128, 128, 3, 3), (18, 384, 1, 1), (256,), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[0],
            end=[32],
            step=[1],
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
            training=[True, True, True, True],
            momentum=[0.01, 0.01, 0.01, 0.01],
            eps=[0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(43458, 64, 32), (32855, 64, 32), (6, 256, 62, 54), (2, 256, 62, 54)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(64,), (64,), (256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(64,), (64,), (256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'bitwise_and': dict(
        name=["bitwise_and"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(642816,), (107136,), (1928448,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(642816,), (107136,), (1928448,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 2, 0, 0, 1, 1, 1, 1, 5, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(107061,), (107077,), (106922,)], [(31440, 32, 4), (31440, 32, 3), (31440, 32, 3)], [(6553, 4), (5551, 4), (6094, 4), (4820, 4), (3529, 4), (4377, 4)], [(4092,), (5316,), (6596,), (5011,), (5141,), (4863,)], [(58, 1), (58, 1), (58, 1), (58, 1), (58, 1), (58, 1), (58, 1)], [(63, 1), (63, 1), (63, 1), (63, 1), (63, 1), (63, 1), (63, 1)], [(14, 2), (14, 2)], [(9, 2), (9, 2)], [(216, 248, 1, 1, 2, 1), (216, 248, 1, 1, 2, 1), (216, 248, 1, 1, 2, 1), (216, 248, 1, 1, 2, 3), (216, 248, 1, 1, 2, 1)], [(1, 248, 216, 3, 2, 7)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(36864,), (64,), (64,), (36864,), (64,), (64,), (36864,), (64,), (64,), (36864,), (64,), (64,), (73728,), (128,), (128,), (147456,), (128,), (128,), (147456,), (128,), (128,), (147456,), (128,), (128,), (147456,), (128,), (128,), (147456,), (128,), (128,), (294912,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (8192,), (128,), (128,), (65536,), (128,), (128,), (524288,), (128,), (128,), (6912,), (18,), (16128,), (42,), (4608,), (12,), (64,), (64,), (640,), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (128,), (128,), (128,), (128,), (128,), (128,), (64,), (64,)]],
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
        para=dict(
            min=[0, 0, 0, 0, None],
            max=[1, 1, None, None, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(28,), (80,), (13, 107136, 2), (16, 107136, 2), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        interface=["torch.nn.functional"],
        para=dict(
            bias=[None, None, None, None, None, None],
            stride=[(4, 4), (1, 1), (4, 4), (1, 1), (2, 2), (2, 2)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            output_padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            groups=[1, 1, 1, 1, 1, 1],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 62, 54), (2, 64, 248, 216), (6, 256, 62, 54), (6, 64, 248, 216), (6, 128, 124, 108), (2, 128, 124, 108)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 128, 4, 4), (64, 128, 1, 1), (256, 128, 4, 4), (64, 128, 1, 1), (128, 128, 2, 2), (128, 128, 2, 2)],
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (2, 2), (1, 1), (1, 1), (2, 2)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 384, 248, 216), (2, 384, 248, 216), (2, 384, 248, 216), (6, 384, 248, 216), (6, 384, 248, 216), (6, 384, 248, 216), (2, 128, 124, 108), (2, 128, 124, 108), (6, 128, 124, 108), (6, 128, 124, 108), (2, 256, 62, 54), (6, 64, 248, 216), (6, 64, 248, 216), (2, 64, 496, 432), (2, 64, 248, 216), (2, 64, 248, 216), (6, 256, 62, 54), (6, 64, 496, 432)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(42, 384, 1, 1), (18, 384, 1, 1), (12, 384, 1, 1), (12, 384, 1, 1), (42, 384, 1, 1), (18, 384, 1, 1), (256, 128, 3, 3), (128, 128, 3, 3), (256, 128, 3, 3), (128, 128, 3, 3), (256, 256, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (64, 64, 3, 3), (256, 256, 3, 3), (64, 64, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(42,), (18,), (12,), (12,), (42,), (18,), None, None, None, None, None, None, None, None, None, None, None, None],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cos': dict(
        name=["cos"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(802, 1), (858, 1)],
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
                    "shape": [(40337, 1, 3), (33456, 1, 3), (37, 1), (81, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(40337, 1, 1), (33456, 1, 1), (37, 1), (81, 1)],
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
            other=[0.95171, 0.999891, 0.99959, 0.999546, 0.897592, 0.563423],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(12,), (18,), (256, 128, 4, 4), (18, 384, 1, 1), (64, 10), (64, 10)],
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
            other=[3.14159, 3.14159, 892, 878, 0.111111, 0.111111],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(125,), (88,), (), (), (791, 7), (843, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(107136,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'eq_case_2': dict(
        name=["eq"],
        interface=["torch"],
        para=dict(
            other=[1, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(41658,), (36422,)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 0, 0, 0, 0, 0, 0, 0, 8, 9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(37065, 64), (8866, 4), (34318, 32, 64), (34190, 32, 64), (64, 128, 1, 1), (256, 128, 4, 4), (42,), (64,), (), ()],
                    "dtype": [np.int32],
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
                    "shape": [(108,), (103,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0.6, 0, 0.5, 0, 0, 0.45, 0.35],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(107136,), (107136,), (107136,), (642816,), (1928448,), (), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(36985, 1), (37747, 1)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 32), (1, 32)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'gt_case_2': dict(
        name=["gt"],
        interface=["torch"],
        para=dict(
            other=[0.785398, 0.785398],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2,), (15,)],
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
                    "requires_grad": [True],
                    "shape": [(40768, 64), (8518, 3), (9,), (6,)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(40768,), (3,), (18,), (13,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'index_put': dict(
        name=["index_put"],
        interface=["CustomizedTest"],
        para=dict(
            accumulate=[True, True, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(36735, 64), (34586, 64), (107136,), (107136,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(36735,), (34586,), (106843,), (33,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(7450, 64), (8471, 64), (), (33,)],
                    "dtype": [np.int64],
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
        para=dict(
            bias=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(36292, 32, 10), (42754, 32, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 10), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'linspace': dict(
        name=["linspace"],
        interface=["torch"],
        para=dict(
            start=[-0.6, 0, -1.78, -39.68],
            end=[-0.6, 69.12, -1.78, 39.68],
            steps=[2, 217, 2, 249],
        ),
    ),

    'log': dict(
        name=["log"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(75, 1), (87, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
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
                    "shape": [(847, 2), (843, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'lt': dict(
        name=["lt"],
        interface=["torch"],
        para=dict(
            other=[0.111111, 0.111111, 0.45, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(876, 7), (831, 7), (107136,), (642816,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1],
            keepdim=[True, True, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32824, 32, 64), (33963, 32, 64), (11, 107136), (16, 107136)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max_case_2': dict(
        name=["max"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1928448,), (642816,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'maximum': dict(
        name=["maximum"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 107136), (14, 107136), (2, 1, 2), (12, 1, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1,), (1,), (1, 107136, 2), (1, 107136, 2)],
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
            dim=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 128, 3, 3), (42, 384, 1, 1), (64, 128, 1, 1), (12,), (64, 64, 3, 3), (256, 128, 4, 4), (18,), (42,), (), (128,), (128, 64, 3, 3), (256,), (256, 256, 3, 3), (128, 128, 2, 2), (12, 384, 1, 1), (64,), (256, 128, 3, 3), (18, 384, 1, 1), (64, 10)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'minimum': dict(
        name=["minimum"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(6, 1, 2), (5, 1, 2), (9, 1, 2), (8, 1, 2), (16, 1, 2), (1, 1, 2), (7, 1, 2), (10, 1, 2), (2, 1, 2), (15, 1, 2), (4, 1, 2), (14, 1, 2), (11, 1, 2), (13, 1, 2), (17, 1, 2), (12, 1, 2), (3, 1, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2), (1, 107136, 2)],
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
                    "shape": [(4271, 3), (5898, 3), (901,), (735,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3,), (3,), (901,), (735,)],
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
                    "shape": [(34450, 32, 10), (44277, 32, 10), (42, 384, 1, 1), (64, 128, 1, 1), (256,), (128,), (1928448, 3), (642816, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(34450, 32, 1), (44277, 32, 1), (), (), (), (), (1928448, 3), (642816, 3)],
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
            other=[0.99, 0.949868, 0.949906, 0.949972, 0.949905, 0.94995],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(18, 384, 1, 1), (256, 128, 3, 3), (12,), (64,), (64, 10), (64, 10)],
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
            other=[432, 432, 0.5, 4, 0.2, 0.2, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(5338,), (8779,), (893, 7), (33833, 1), (), (), (12, 384, 1, 1), (18, 384, 1, 1)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(817, 2), (794, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "requires_grad": [False],
                    "shape": [(817,), (794,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(849, 1), (843, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'nonzero': dict(
        name=["nonzero"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(33988,), (40461,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            dim=[(0, 1, 2, 3), (0,), (0, 1), (0,), (0, 1, 2, 3), (0,), (0,), (0,), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0,), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3), (0,), (0, 1, 2, 3)],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(12, 384, 1, 1), (12,), (64, 10), (256,), (256, 128, 3, 3), (18,), (42,), (128,), (128, 128, 3, 3), (128, 128, 2, 2), (256, 256, 3, 3), (64,), (64, 64, 3, 3), (18, 384, 1, 1), (128, 64, 3, 3), (64, 128, 1, 1), (42, 384, 1, 1), (66,), (256, 128, 4, 4)],
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
            mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.01, 0.176777, 0.01, 0.0416667, 0.0625, 0.0416667, 0.0294628, 0.01, 0.0220971, 0.0589256, 0.0294628],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(12, 384, 1, 1), (64, 128, 1, 1), (18, 384, 1, 1), (128, 128, 3, 3), (128, 128, 2, 2), (128, 64, 3, 3), (256, 128, 3, 3), (42, 384, 1, 1), (256, 128, 4, 4), (64, 64, 3, 3), (256, 256, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'pow': dict(
        name=["pow"],
        interface=["torch"],
        para=dict(
            exponent=[2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(113, 1), (77, 1)],
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

    'relu': dict(
        name=["relu"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(31963, 32, 64), (32071, 32, 64)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'relu_case_2': dict(
        name=["relu"],
        interface=["torch.nn.functional"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6, 128, 248, 216), (2, 128, 124, 108), (6, 64, 248, 216), (2, 64, 248, 216), (2, 128, 248, 216), (6, 256, 62, 54), (2, 256, 62, 54), (6, 128, 124, 108)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'repeat': dict(
        name=["repeat"],
        interface=["torch.Tensor"],
        para=dict(
            repeats=[(1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (216, 248, 1, 1, 2, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(216, 248, 1, 1, 2), (216, 248, 1, 1, 2), (216, 248, 1, 1, 2), (216, 248, 1, 1, 2), (1, 1, 1, 1, 1, 3)],
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
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(31735, 32, 64), (36985, 32, 64)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["index"],
                    "requires_grad": [False],
                    "shape": [(31735, 1, 64), (36985, 1, 64)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", high=32),
                },
                {
                    "ins": ["src"],
                    "requires_grad": [False],
                    "shape": [(31735, 1, 64), (36985, 1, 64)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sgn': dict(
        name=["sgn"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(888, 7), (815, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sin': dict(
        name=["sin"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(817, 1), (836, 1)],
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
                    "shape": [(131, 1), (77, 1), (12, 384, 1, 1), (256, 256, 3, 3), (12,), (256,)],
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
                    "shape": [[(321408, 7), (321408, 7), (321408, 7), (321408, 7), (321408, 7), (321408, 7)], [(64, 214272), (64, 214272), (64, 214272), (64, 214272), (64, 214272), (64, 214272)], [(321408, 7), (321408, 7)], [(321408,), (321408,)], [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]],
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
            alpha=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(38168, 32, 3), (32729, 32, 3), (33009, 32), (41131, 32), (29,), (18,), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(38168, 1, 3), (32729, 1, 3), (33009, 1), (41131, 1), (29,), (18,), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sub_case_2': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            other=[-1.5708, -1.5708, 0.0555556, 0.0555556, 1],
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(118,), (116,), (841, 7), (884, 7), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[None, None, [1], [1], None, None],
            keepdim=[False, False, True, True, False, False],
            dtype=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(824, 7), (869, 7), (42039, 32, 3), (38590, 32, 3), (809,), (798,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'unique': dict(
        name=["unique"],
        interface=["torch"],
        para=dict(
            sorted=[True, True],
            return_inverse=[False, False],
            return_counts=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(84,), (54,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(799, 7), (909, 7)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(799, 7), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(), (909, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
