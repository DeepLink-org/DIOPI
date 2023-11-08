import numpy as np

retinanet_config = {
    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(201600, 4), (69552, 4)],
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
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(492,), (206,), (24, 1), (30, 1), (1, 256, 50, 68), (1, 256, 100, 136), (1, 9, 4), (1, 9, 4), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(492,), (206,), (1, 169371), (1, 161145), (1, 256, 50, 68), (1, 256, 100, 136), (966, 1, 4), (3500, 1, 4), ()],
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
            alpha=[1, 1, -0.00333333, -0.00159158],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 1024, 50, 70), (2, 1024, 72, 50), (1024,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 1024, 50, 70), (2, 1024, 72, 50), (1024,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 0],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(280,), (240,), ()],
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
                    "shape": [(172773,), (139284,)],
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
            end=[60, 136, 46, 62, 164, 160, 80, 152, 100, 35, 24, 34, 41, 26, 124, 112, 66, 30, 128, 76, 7, 32, 6, 29, 15, 148, 33, 140, 116, 104, 10, 18, 37, 68, 58, 16, 50, 12, 11, 64, 144, 8, 14, 17, 156, 74, 22, 88, 72, 120, 70, 23, 21, 39, 54, 20, 13, 5, 40, 36, 132, 168, 25, 31, 9, 48, 28, 42, 56, 92, 44, 82, 78, 108, 96, 19, 27, 52, 84, 38, 4, 3],
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
                    "shape": [(1, 512, 168, 100), (2, 128, 100, 104)],
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
                    "shape": [(2900,), (850,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(2900,), (850,)],
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
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(140400,), (35100,), (8775,), (2340,), (630,)], [(144000,), (36000,), (9000,), (2340,), (630,)], [(9408,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (131072,), (256,), (262144,), (256,), (524288,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (4718592,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (1658880,), (720,), (82944,), (36,), (3,), (3,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
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
        para=dict(
            min=[0, 0],
            max=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(27, 191970, 2), (32, 104913, 2)],
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
            padding=[(1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 128, 96, 336), (1, 256, 7, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(128, 128, 3, 3), (36, 256, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [None, (36,)],
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
                    "shape": [(3, 971, 800), (3, 968, 800), (17,), (230,), (17, 185436), (3, 88749)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (17,), (230,), (17, 185436), (3, 88749)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(666, 4), (989, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4)],
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
            other=[750, 1165, 33, 1138, 720, 712, 325, 1316, 896, 769],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), (), (), (), (), (), ()],
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
                    "shape": [(168048,), (120087,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(), ()],
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
                    "shape": [(187245,), (158481,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[-1, -1, 0, 0, 0, 0, 38, 26],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(191970,), (144009,), (3, 896, 800), (3, 704, 1344), (168048, 4), (12150, 80), (), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (148851,), (196875,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0.05, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(112851,), (196875,), (8325, 80), (22500, 80)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(20, 4), (17, 4), (20,), (120978,), (3, 800, 1088), (3, 568, 1333)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(514,), (249,), (436,), (52,), (3,), (3,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-3, high=3),
                },
            ],
        ),
    ),

    'index_put': dict(
        name=["index_put"],
        interface=["CustomizedTest"],
        para=dict(
            accumulate=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(196875,), (177678,), (169371, 4), (128916, 4)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(196875,), (177678,), (169371, 4), (128916,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(), (), (), ()],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
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
                    "shape": [(840,), (480,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'lt': dict(
        name=["lt"],
        interface=["torch"],
        para=dict(
            other=[0.4, 0.4],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(120978,), (112851,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[0, 0],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(77, 120087), (14, 177309)],
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
                    "shape": [(1, 64, 576, 400), (1, 64, 496, 400)],
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
                    "requires_grad": [False],
                    "shape": [(18, 177678), (3, 196875), (49, 1, 2), (5, 1, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1,), (1,), (1, 153576, 2), (1, 134379, 2)],
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
            dim=[None, None, None, None, None],
            keepdim=[False, False, False, False, False],
            dtype=[None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(36, 256, 3, 3), (128, 512, 1, 1), (), (720,), (64,)],
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
                    "shape": [(61, 1, 2), (47, 1, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 182403, 2), (1, 163206, 2)],
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
                    "shape": [(16, 163206), (11250, 80), (37,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(16, 163206), (11250, 1), (37,), (64,)],
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
                    "shape": [(3510, 80), (1188, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3510, 80), (1188, 80)],
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
            other=[0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 512, 1, 1), (720, 256, 3, 3), (720,), (36,)],
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
            other=[0.5, 0.5, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4,), (640,), (512, 128, 1, 1), (128, 128, 3, 3), (), ()],
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
                    "shape": [(182403,), (148851,), (102816, 80), (108864, 80)],
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
            mean=[0, 0, 0],
            std=[0.01, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(720, 256, 3, 3), (256, 256, 3, 3), (36, 256, 3, 3)],
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
                    "shape": [(1, 256, 136, 336), (1, 128, 52, 168)],
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
            repeats=[(50,), (50,), (1, 72), (1, 100)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(76,), (54,), (50, 1), (112, 1)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(4158, 4), (63504, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "requires_grad": [False],
                    "shape": [(2079, 80), (1134, 80)],
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

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(170,), (170,), (170,), (170,)], [(1124,), (1124,), (1124,), (1124,)], [(153207,), (153207,)], [(134379,), (134379,)], [(3, 1248, 800)], [(3, 1344, 704)]],
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
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(317,), (529,), (56, 163206, 2), (3, 1197, 800), (36, 193374), (49, 153576)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(317,), (529,), (56, 163206, 2), (3, 1, 1), (36, 193374), (49, 153576)],
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
                    "shape": [(995, 4), (29, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4)],
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
                    "shape": [(343,), (83,), ()],
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
            dim=[None, None],
            keepdim=[False, False],
            dtype=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(57456, 4), (4446, 4)],
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
            start=[-0.0883883, -0.0170103, -0.051031, -0.0360844, -0.0684653],
            end=[0.0883883, 0.0170103, 0.051031, 0.0360844, 0.0684653],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 512, 1, 1), (256, 2048, 3, 3), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 1024, 1, 1)],
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
                    "shape": [(176080,), (171894,)],
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
            size=[(108, 100), (54, 50)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 256, 54, 50), (1, 256, 27, 25)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
