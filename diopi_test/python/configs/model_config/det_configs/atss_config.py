import numpy as np

atss_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(13343, 1), (285, 4), (383,), (33,), (1, 256, 66, 50), (1, 256, 30, 84), (1, 1, 4), (1, 1, 4), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 22), (1, 4), (383,), (33,), (1, 256, 66, 50), (1, 256, 30, 84), (33, 1, 4), (44, 1, 4), (), ()],
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
            alpha=[1, 1, -0.00519519, -0.00463463, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 100, 164), (1, 256, 320, 200), (), (), (334,), (233,), (215, 4), (168, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 100, 164), (1, 256, 320, 200), (), (), (334,), (233,), (215, 4), (168, 2)],
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
            other=[226831, 20604, 211167, 288018, 749879, 644580, 848906, 66156, 80005, 140872],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,)],
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
            other=[17000, 13650, 1, 1, 0, 0],
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(9, 54), (9, 25), (451,), (328,), (), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'all': dict(
        name=["all"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(0,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'any': dict(
        name=["any"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(328,), (121,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            end=[14, 100, 70, 36, 37, 24, 68, 40, 48, 160, 34, 108, 124, 56, 8, 54, 7, 26, 52, 5, 60, 116, 152, 13, 82, 30, 136, 33, 12, 72, 120, 42, 168, 128, 38, 31, 76, 18, 132, 9, 11, 17, 32, 27, 39, 25, 20, 41, 80, 6, 66, 29, 62, 50, 64, 96, 104, 84, 35, 164, 148, 28, 19, 58, 10, 140, 74, 78, 15, 156, 144, 112, 21, 16, 44, 88, 22, 4, 46, 92, 23, 3],
            step=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
                    "requires_grad": [True],
                    "shape": [(1, 512, 18, 42), (2, 128, 200, 264)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(512,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(512,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(512,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(512,), (128,)],
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
                    "shape": [(252,), (302,), (45, 9), (45, 5)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(252,), (302,), (45, 9), (45, 5)],
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
            dim=[1, 1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(215, 2), (215, 2)], [(144, 2), (144, 2)], [(10800,), (2700,), (675,), (182,), (49,)], [(9, 17), (9, 17), (9, 17), (9, 17), (9, 17)], [(9408,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (131072,), (256,), (262144,), (256,), (524288,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (184320,), (80,), (9216,), (4,), (2304,), (1,), (1,), (1,), (1,), (1,), (1,), (3,), (3,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
            min=[1],
            max=[None],
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

    'clamp_min': dict(
        name=["clamp_min"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(186,), (343,)],
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
            min=[0, 0, -4.13517, 0],
            max=[None, None, 4.13517, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(15238, 3, 2), (21486, 5, 2), (428, 2), (46, 2)],
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
            stride=[(1, 1), (1, 1)],
            padding=[(0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 1024, 50, 76), (1, 256, 50, 68)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 1024, 1, 1), (256, 256, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,)],
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
                    "shape": [(3, 1227, 800), (3, 1191, 800), (112,), (342,), (17023, 32), (18672, 42)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (112,), (342,), (17023, 32), (18672, 42)],
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
            other=[1],
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

    'div_case_3': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[81.1753, 58.782, 2, 2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (35,), (248,), (292, 2), (10, 2)],
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
                    "shape": [(257,), (80,), (19, 2), (300, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (1,), (19, 2), (300, 2)],
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
            other=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(19197,), (16001,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'exp': dict(
        name=["exp"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(77, 2), (218, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'exp_case_2': dict(
        name=["exp"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(87,), (40,)],
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
            value=[-100000000, 0, 1, 1, 0, 0, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14931, 20), (94, 4), (42,), (248,), (3, 512, 1344), (3, 640, 1344), (), ()],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 20), (45, 13)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 20), (1, 13)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'ge_case_2': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(80, 2), (305, 2), (113,), (26400,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=["group_norm"],
        interface=["torch"],
        para=dict(
            N=[1, 2],
            C=[256, 256],
            HxW=[8736, 798],
            group=[32, 32],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 256, 52, 168), (2, 256, 19, 42)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(328, 2), (121, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(328, 2), (121, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'gt_case_2': dict(
        name=["gt"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256,), (241,), (45, 49), (45, 10)],
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
                    "shape": [(18672, 1), (756, 4), (40534,), (96720,), (3, 1259, 800), (3, 402, 1333)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(45, 1), (0,), (26,), (270,), (3,), (3,)],
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
            accumulate=[False, True, True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(22400, 4), (31200, 4), (1950,), (871481,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(22400,), (209,), (82,), (422,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(14406, 4), (209, 4), (82,), (422,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'isnan': dict(
        name=["isnan"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(44,), (276,)],
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
            other=[4.13517, 4.13517],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(352, 2), (288, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'log': dict(
        name=["log"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(120,), (189,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
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
                    "shape": [(101, 2), (342, 2)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "shape": [(101, 2), (342, 2)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'lt': dict(
        name=["lt"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(172,), (233,), (231, 2), (144, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (1,), (231, 2), (144, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'lt_case_2': dict(
        name=["lt"],
        interface=["torch"],
        para=dict(
            other=[80, 80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(24800,), (6800,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(75, 2), (48, 2), (325,), (361,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "requires_grad": [False],
                    "shape": [(75, 2), (48, 2), (325,), (361,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(13868, 6), (332, 2)],
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
            kernel_size=[(3, 3), (3, 3)],
            stride=[(2, 2), (2, 2)],
            padding=[(1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            ceil_mode=[False, False],
            return_indices=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 64, 400, 560), (1, 64, 656, 400)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    # "requires_grad": [True],
                    "shape": [(281,), (370,), (8958, 1, 2), (14406, 1, 2), (194, 2), (21330, 17)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1,), (1,), (1, 9, 2), (1, 4, 2), (194, 2), (1,)],
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
            dim=[[0], [0], None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 25), (45, 60), (256,), (2048,), (256, 512, 1, 1), (1024, 256, 1, 1), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'min': dict(
        name=["min"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1],
            keepdim=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(63, 2), (140, 2), (45, 4, 26), (45, 4, 42)],
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
                    # "requires_grad": [True],
                    "shape": [(7162, 1, 2), (20604, 1, 2), (312, 2), (319, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 4, 2), (1, 31, 2), (312, 2), (319, 2)],
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
                    "shape": [(1, 4, 21, 13), (2, 4, 28, 25), (100,), (44,), (88, 4), (77, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (100,), (44,), (1, 4), (1, 4)],
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
                    "shape": [(150,), (85,), (32800, 80), (390, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(150,), (85,), (32800, 80), (390, 80)],
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
            other=[0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (1024,), (512, 512, 3, 3), (128, 256, 1, 1), ()],
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
            other=[0.5, 0.5, 1, 1, 16, 16, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(93, 2), (243, 2), (512, 2048, 1, 1), (4, 256, 3, 3), (84,), (78,), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch"],
        para=dict(
            other=[-100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(155,), (66,)],
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
                    "shape": [(207, 2), (330, 2), (168,), (7,)],
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
                    "shape": [(28224,), (1485,), (950, 80), (231, 80)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
            std=[0.01, 0.01, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 256, 3, 3), (256, 256, 3, 3), (80, 256, 3, 3), (1, 256, 3, 3)],
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
                    "shape": [(22400, 19, 2), (21486, 47, 2)],
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
                    "shape": [(2, 512, 100, 132), (2, 128, 100, 104)],
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
            repeats=[(1, 1), (1, 168), (100,), (88,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(290, 4), (44, 1), (112,), (168,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=["torch"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(94,), (419,), (7392, 80), (875, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=["torch"],
        para=dict(
            dim=[-1],
            descending=[True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(0,)],
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
                    "shape": [(20267, 60), (16539, 30), (300,), (340,)],
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
            dim=[1, 1, 1, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(44,), (44,), (44,), (44,)], [(45, 57), (45, 57), (45, 57), (45, 57)], [(331,), (331,)], [(13442,), (13442,)], [(3, 800, 1184)], [(3, 1312, 800)]],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'std': dict(
        name=["std"],
        interface=["torch"],
        para=dict(
            dim=[[0], [0]],
            correction=[1, 1],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(45, 15), (45, 20)],
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
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(18819, 7, 2), (3, 800, 920), (122, 2), (49, 2), (227,), (303,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(18819, 7, 2), (3, 1, 1), (122, 2), (49, 2), (227,), (303,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sub_case_2': dict(
        name=["sub"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(366,), (548,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(366,), (548,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sub_case_3': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(405,), (228,), ()],
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
            dim=[[-1], [-1], None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(21486, 28, 2), (13868, 26, 2), (283,), (72,), (8064, 80), (5000, 80), (2, 4, 100, 100), (2, 4, 164, 100), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch"],
        para=dict(
            k=[9, 9],
            dim=[0, 0],
            largest=[False, False],
            sorted=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(12768, 4), (3200, 31)],
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
            start=[-0.0684653, -0.051031, -0.0360844, -0.0883883],
            end=[0.0684653, 0.051031, 0.0360844, 0.0883883],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 512, 1, 1)],
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
                    "shape": [(332,), (17807,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        interface=["torch.nn.functional"],
        para=dict(
            mode=['nearest', 'nearest'],
            size=[(124, 100), (100, 160)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 256, 62, 50), (2, 256, 50, 80)],
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
                    "shape": [(104, 2), (238, 2), (5,), (270,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(104, 2), (238, 2), (5,), (270,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(104, 2), (238, 2), (5,), (270,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
