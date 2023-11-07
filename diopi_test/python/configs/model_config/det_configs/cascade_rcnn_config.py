import numpy as np

cascade_rcnn_config = {
    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(58, 4), (97, 4)],
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
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(48, 1), (53, 1), (24,), (31,), (1, 3, 4), (1, 3, 4), (2, 256, 304, 200), (1, 256, 100, 120), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 191769), (1, 226128), (24,), (31,), (875, 1, 4), (59200, 1, 4), (2, 256, 304, 200), (1, 256, 100, 120), (), ()],
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
            alpha=[1, 1, -0.0014214, -0.00478477, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 29, 25), (2, 256, 100, 148), (4,), (512,), (6, 1024), (59, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 29, 25), (2, 256, 100, 148), (4,), (512,), (6, 1024), (59, 4)],
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
            other=[1e-06, 1, 0],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(40,), (9,), ()],
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
                    "shape": [(4378,), (4780,)],
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
                    "shape": [(60,), (37,)],
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
            start=[1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            end=[65, 24, 42, 46, 5, 6, 21, 21, 34, 124, 53, 35, 20, 296, 10, 60, 16, 24, 29, 14, 31, 9, 288, 28, 328, 304, 60, 52, 2, 264, 168, 12, 47, 336, 38, 55, 48, 140, 4, 26, 52, 70, 18, 96, 26, 31, 56, 108, 78, 43, 39, 45, 18, 192, 34, 17, 17, 49, 41, 27, 32, 40, 56, 58, 25, 160, 148, 59, 19, 13, 36, 3, 14, 28, 46, 41, 80, 44, 208, 27, 216, 23, 152, 51, 8, 116, 256, 64, 37, 33, 76, 35, 74, 22, 248, 112, 100, 144, 32, 68, 48, 312, 30, 62, 29, 136, 20, 54, 42, 164, 104, 33, 224, 272, 128, 13, 11, 36, 184, 54, 11, 82, 200, 15, 132, 92, 25, 23, 240, 30, 84, 58, 320, 280, 16, 15, 7, 66, 72, 120, 37, 12, 50, 156, 19, 40, 38, 39, 50, 232, 66, 69, 176, 62, 22, 91, 78, 57, 75, 63, 68, 88, 10, 76, 67, 61, 44, 64, 73, 70, 7, 8, 6, 9],
            step=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
    ),

    'argmax': dict(
        name=["argmax"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(31, 80), (45, 80)],
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
                    "requires_grad": [True],
                    "shape": [(2, 512, 168, 100), (2, 128, 200, 224)],
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
                    "shape": [(164197,), (120673,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(164197,), (120673,)],
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
            dim=[1, 0, 0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(8, 1), (8, 4)], [(4,), (24,)], [(1000, 4), (1000, 4), (1000, 4), (1000, 4), (663, 4)], [(1000, 4), (1000, 4), (1000, 4), (1000, 4), (546, 4)], [(1024,), (1048576,), (1024,), (3,), (3,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(9408,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (65536,), (256,), (131072,), (256,), (262144,), (256,), (524288,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (768,), (3,), (3072,), (12,), (82944,), (81,), (4096,), (4,), (12845056,), (1024,), (1048576,), (1024,), (82944,), (81,), (4096,), (4,), (12845056,), (1024,), (1048576,), (1024,), (82944,), (81,), (4096,), (4,), (12845056,)], [(0, 5)]],
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
            max=[1020, 1151],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 2), (42, 2)],
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
                    "shape": [(64800, 1), (306432, 1)],
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
            min=[0, 0, 0, 0, -4.13517, -4.13517],
            max=[None, None, 3, 3, 4.13517, 4.13517],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(17, 226128, 2), (3, 218865, 2), (3,), (65,), (49, 2), (59, 2)],
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
                    "shape": [(1, 256, 336, 152), (2, 256, 42, 24)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(12, 256, 1, 1), (256, 256, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(12,), (256,)],
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
                    "shape": [(16, 149909), (2, 185754), (3, 1138, 800), (3, 725, 1333), (26,), (44,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(16, 149909), (2, 185754), (3, 1, 1), (3, 1, 1), (26,), (44,)],
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
                    "shape": [(9, 4), (72, 4)],
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
            other=[1, 1, 76, 2, 56, 56],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(122, 4), (115, 4), (), (), (26,), (74,)],
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
                    "shape": [(157568,), (144128,), (1, 51), (1, 38)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (1, 51), (1, 38)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(43,), (236148,)],
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
                    "requires_grad": [False],
                    "shape": [(11, 2), (35, 2)],
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
                    "shape": [(1512, 1), (15000, 1)],
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
            value=[0, 0, 0, 0, 0, 0, 0, 0, 5, 41],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(0,), (12,), (105223, 4), (184610, 4), (250, 256, 7, 7), (610, 256, 7, 7), (3, 1088, 800), (3, 544, 1344), (), ()],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(23,), (59,)],
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
            other=[0.7, 0, 0.3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(195317,), (116495,), ()],
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
            other=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(182450,), (488,)],
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
                    "shape": [(182337,), (218574,), (8631, 4), (120673, 4), (3, 726, 1333), (3, 806, 800), (62, 256, 7, 7), (60, 256, 7, 7)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(152,), (250,), (2904,), (255,), (3,), (3,), (48,), (48,)],
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
            accumulate=[False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(107156,), (498,), (143029, 4), (149902, 4), (29, 256, 7, 7), (39, 256, 7, 7)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(107156,), (498,), (13,), (10,), (6,), (17,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-29, high=29),
                },
                {
                    "ins": ["values"],
                    # "requires_grad": [True],
                    "shape": [(), (12,), (), (), (6, 256, 7, 7), (17, 256, 7, 7)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(58, 12544), (103, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1024, 12544), (81, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1024,), (81,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(41,), (9,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'log2': dict(
        name=["log2"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(17,), (23,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'log_case_2': dict(
        name=["log"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(21000, 1), (22200, 1)],
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
                    "shape": [(78, 81), (40, 81)],
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
            other=[0.3, 0.3, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(198510,), (220202,), (92, 4), (121, 4)],
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
            dim=[0, 1],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 191009), (3, 148676)],
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
                    "shape": [(8539, 4), (8624, 4)],
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
                    "shape": [(1, 64, 240, 672), (2, 64, 400, 656)],
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
                    "shape": [(45, 1, 2), (7, 1, 2), (17, 234639), (2, 236739)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 195063, 2), (1, 192681, 2), (1,), (1,)],
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
            dim=[None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (12,), (1024, 256, 1, 1), (2048, 512, 1, 1), (1024, 12544), (4, 1024), ()],
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
                    "shape": [(40, 1, 2), (37, 1, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 218865, 2), (1, 216139, 2)],
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
                    "shape": [(13, 227138), (9, 201856), (32,), (192815,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(13, 227138), (9, 201856), (32,), (192815,)],
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
                    "shape": [(25200, 1), (1170, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(25200, 1), (1170, 1)],
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
            other=[1.35135, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (512,), (1024, 256, 1, 1), (512, 512, 3, 3), (4, 1024), (81, 1024)],
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
            other=[0.5, 0.5, 0.5, 0.5, 1, 1, 0, 0.25],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(11,), (61,), (62400, 4), (100, 4), (512, 2048, 1, 1), (512, 128, 1, 1), (), ()],
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
                    "shape": [(42, 81), (64, 81)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "requires_grad": [False],
                    "shape": [(42,), (64,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(64800,), (1092,)],
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
                    "shape": [(4500, 1), (24600, 1)],
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
                    "shape": [(157159,), (232089,)],
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
            mean=[0, 0, 0, 0, 0],
            std=[0.01, 0.01, 0.01, 0.01, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(81, 1024), (3, 256, 1, 1), (256, 256, 3, 3), (12, 256, 1, 1), (4, 1024)],
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
                    "shape": [(1, 256, 100, 124), (1, 256, 28, 25)],
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
                    "shape": [(1, 512, 25, 29), (2, 512, 50, 58), (71, 1024), (28, 1024)],
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
            repeats=[(240,), (200,), (1, 100), (1, 54)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(200,), (264,), (136, 1), (50, 1)],
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
                    "shape": [(64, 4), (134, 4)],
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
                    "shape": [(2520, 1), (1950, 1)],
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
            dim=[-1, -1],
            descending=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(134400,), (26208,)],
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
                    "shape": [(69,), (41,)],
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
            dim=[0, 0, 1, 1, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(247086,), (247086,)], [(172671,), (172671,)], [(16800,), (16800,), (16800,), (16800,)], [(6,), (6,), (6,), (6,)], [(3, 960, 800)], [(3, 928, 800)]],
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
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 175933, 2), (5, 241557, 2), (11, 226128), (44, 229938), (87,), (119,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 175933, 2), (5, 241557, 2), (11, 226128), (44, 229938), (87,), (119,)],
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
                    "shape": [(50, 4), (55, 4)],
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
            other=[1, 1, 0.0555556, 0.5, 1],
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(55,), (145,), (19200, 4), (202, 4), ()],
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
            dim=[None, None, None, None, [0], None],
            keepdim=[False, False, False, False, True, False],
            dtype=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 50, 58), (2, 256, 136, 100), (71, 4), (59, 4), (7,), (75,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch"],
        para=dict(
            k=[1, 1],
            dim=[1, 1],
            largest=[True, True],
            sorted=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(25, 81), (45, 81)],
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
            start=[-0.0541266, -0.108253, -0.0360844, -0.0684653, -0.021029, -0.0883883, -0.051031],
            end=[0.0541266, 0.108253, 0.0360844, 0.0684653, 0.021029, 0.0883883, 0.051031],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 1024), (256, 256, 1, 1), (256, 256, 3, 3), (256, 1024, 1, 1), (1024, 12544), (256, 512, 1, 1), (256, 2048, 1, 1)],
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
            sorted=[True, True, True],
            return_inverse=[False, False, False],
            return_counts=[False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(12,), (65,), (0, 1)],
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
            size=[(66, 50), (50, 78)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 256, 33, 25), (1, 256, 25, 39)],
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
                    "shape": [(1014, 4), (81, 4), (33,), (76,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1014, 4), (81, 4), (33,), (76,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(), (), (33,), (76,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

}
