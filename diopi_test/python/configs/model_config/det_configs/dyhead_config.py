import numpy as np

dyhead_config = {
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
                    "shape": [(1, 256, 144, 240), (1, 256, 24, 24)],
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
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(157, 2), (34100, 1), (107,), (154,), (1, 256, 160, 208), (1, 256, 72, 104), (1, 133120, 192), (1, 1280, 768), (1, 128, 12, 144, 144), (1, 660, 6, 144, 144), (), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(157, 2), (1, 7), (107,), (154,), (1, 256, 1, 1), (1, 256, 72, 104), (1, 133120, 192), (1, 1280, 768), (1, 128, 1, 144, 144), (1, 660, 1, 144, 144), (), ()],
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
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 256, 24, 14), (1, 27, 112, 64), (48,), (15,), (192, 4), (157, 2), (1, 4608, 768), (1, 8320, 768), (3, 36, 24, 144, 32), (3, 198, 12, 144, 32), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 256, 24, 14), (1, 27, 112, 64), (48,), (15,), (192, 4), (157, 2), (1, 4608, 768), (1, 8320, 768), (3, 36, 24, 144, 32), (3, 198, 12, 144, 32), ()],
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
            other=[190960, 165726, 572880, 98208, 83545, 917631, 174592, 265980, 143220, 368280],
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
            other=[47600, 40320, 1, 1, 0.904348, 0.852174, 3, 3, 0, 1e-08],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(9, 27), (9, 14), (109,), (267,), (1, 1, 1), (1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (), ()],
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
            value=[-1.90439e-06, -4.17918e-05, -4.08909e-05, -4.21922e-05, -5.37348e-06, -3.28829e-05, -2.007e-05, -2.97798e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(529, 24), (576, 192), (1,), (27,), (), (), (64, 256, 1, 1), (80, 256, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(529, 24), (576, 192), (1,), (27,), (), (), (64, 256, 1, 1), (80, 256, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(529, 24), (576, 192), (1,), (27,), (), (), (64, 256, 1, 1), (80, 256, 1, 1)],
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
            value=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (384, 384), (4,), (3072,), (256, 256, 3, 3), (1024, 64, 1, 1), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (384, 384), (4,), (3072,), (256, 256, 3, 3), (1024, 64, 1, 1), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(768, 768), (384, 384), (4,), (3072,), (256, 256, 3, 3), (1024, 64, 1, 1), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(7,), (1,), (68,), (11,), (6,), (3,), (15,), (12,), (23,), (18,), (4,), (0,), (8,), (22,), (5,), (2,), (43,)],
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
                    "shape": [(83,), (42966,)],
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
            start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            end=[160, 40, 7, 10, 96, 192, 80, 56, 28, 20, 104, 13, 36, 144, 224, 14, 8, 9, 112, 256, 52, 120, 16, 176, 32, 26, 4, 11, 6, 48, 5, 18, 88, 72, 60, 240, 128, 12, 24, 64, 22, 15, 208, 44, 30],
            step=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
                    "shape": [(336,), (2880,), (45, 31), (45, 15)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(336,), (2880,), (45, 31), (45, 15)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(1344, 144, 144), (2508, 144, 32)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mat2"],
                    "requires_grad": [True],
                    "shape": [(1344, 144, 32), (2508, 32, 144)],
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
            dim=[0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(9216,), (192,), (192,), (192,), (192,), (192,), (3174,), (110592,), (576,), (36864,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (192,), (192,), (3174,), (110592,), (576,), (36864,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (768,), (768,), (294912,), (384,), (384,), (6348,), (442368,), (1152,), (147456,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (384,), (6348,), (442368,), (1152,), (147456,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (1536,), (1536,), (1179648,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,)], [(7, 2), (7, 2)], [(61, 2), (61, 2)], [(1,), (0,), (0,), (0,), (0,)], [(1, 4), (0, 4), (0, 4), (0, 4), (0, 4)], [(3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (12696,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,)], [(768,), (3072,), (3072,), (4718592,), (1536,), (1536,), (25392,), (7077888,), (4608,), (2359296,), (1536,), (1536,), (1536,), (9437184,), (6144,), (9437184,), (1536,), (1536,), (1536,), (25392,), (7077888,), (4608,), (2359296,), (1536,), (1536,), (1536,), (9437184,), (6144,), (9437184,), (1536,), (384,), (384,), (768,), (768,), (1536,), (1536,), (98304,), (256,), (196608,), (256,), (393216,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,), (256,), (589824,)], [(1, 256, 1, 1), (1, 256, 1, 1), (1, 256, 1, 1), (1, 256, 1, 1)], [(256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (589824,), (256,), (256,), (589824,), (256,), (256,), (589824,), (256,), (256,), (62208,), (27,), (256,), (1,), (16384,), (64,), (65536,), (1024,), (20480,), (80,), (1024,), (4,), (256,), (1,), (1,), (1,), (1,), (1,), (1,), (3,), (3,)], [(20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,), (20736,)]],
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
            min=[0, 0, 1, 0, 0],
            max=[1, 1, None, 2000, 1200],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 1, 1, 1), (1, 1024, 1, 1), (), (22, 2), (18, 2)],
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
                    "shape": [(151,), (47,)],
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
            min=[0, 0, 0, -4.13517],
            max=[None, None, None, 4.13517],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(38192, 1, 2), (8525, 2, 2), (174, 2), (114, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'col2_im': dict(
        name=["col2_im"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(104, 72), (72, 48)],
            kernel_size=[(2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1)],
            padding=[(0, 0), (0, 0)],
            stride=[(2, 2), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 3072, 1872), (1, 3072, 864)],
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
            padding=[(0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 768, 48, 96), (1, 256, 30, 20)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 768, 1, 1), (256, 256, 3, 3)],
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
                    "shape": [(179,), (10,), (3, 795, 1060), (3, 1104, 1655), (33759, 26), (14322, 16)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(179,), (10,), (3, 1, 1), (3, 1, 1), (33759, 26), (14322, 16)],
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
            other=[0.999173, 0.329674, 0.990238, 0.998269, 0.987306, 0.648765, 0.748458, 0.967255],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(529, 24), (529, 24), (4,), (768,), (4, 256, 1, 1), (1024, 64, 1, 1), (), ()],
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
            other=[141.978, 94.216, 0.930435, 0.921739, 2, 2, 2, 2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (1, 2240, 768), (1, 4224, 768), (1, 256, 16, 8), (1, 256, 80, 112), (179,), (307,), (304, 2), (180, 2)],
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
                    "shape": [(1, 256, 26, 14), (1, 256, 20, 30), (146, 2), (237, 2), (240,), (27,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 256, 26, 14), (1, 256, 20, 30), (146, 2), (237, 2), (1,), (1,)],
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
            other=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(10912,), (39897,), (176, 144, 144), (456, 144, 144)],
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
                    "requires_grad": [True],
                    "shape": [(58, 2), (160, 2)],
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
                    "shape": [(16,), (161,)],
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
            value=[0, 0, 0, 0, -100000000, 0, 0, 0, 0, 0, 1, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 121, 12, 144, 32), (3, 484, 6, 144, 32), (1, 27, 96, 96), (1, 180, 168, 384), (49104, 2), (111, 4), (3, 1664, 1152), (3, 896, 1408), (9,), (17050,), (), ()],
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
                    "shape": [(1, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(45, 23), (45, 36)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 23), (1, 36)],
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
            other=[-4.13517, -4.13517, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(102, 2), (111, 2), (21,), (5,), (1, 1, 1, 1), (1, 1024, 1, 1)],
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
            approximate=['none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 119808, 768), (1, 1008, 6144)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=["group_norm"],
        interface=["torch"],
        para=dict(
            N=[1, 1],
            C=[256, 256],
            HxW=[8960, 640],
            group=[16, 16],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 256, 112, 80), (1, 256, 32, 20)],
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
                    "shape": [(1, 256, 16, 8), (1, 256, 40, 128), (196, 2), (259, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 256, 16, 8), (1, 256, 40, 128), (196, 2), (259, 2)],
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
                    "shape": [(200,), (110,), (45, 77), (45, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'im2_col': dict(
        name=["im2_col"],
        interface=["torch.nn.functional"],
        para=dict(
            kernel_size=[(2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1)],
            padding=[(0, 0), (0, 0)],
            stride=[(2, 2), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1, 768, 32, 128), (1, 384, 64, 256)],
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
                    "shape": [(7,), (65472,), (10230, 42), (19096, 16), (3, 773, 761), (3, 1297, 973)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(71,), (62,), (45, 42), (45, 16), (3,), (3,)],
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
            accumulate=[False, False, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(235290,), (39897,), (384, 4), (1008, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(200,), (113,), (31,), (45,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-384, high=384),
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(200,), (), (31, 4), (45, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'is_nan': dict(
        name=["is_nan"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(228,), (217,)],
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
            normalized_shape=[(192,), (192,)],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 100352, 192), (1, 81920, 192)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(192,), (192,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(192,), (192,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_mean"],
                    "requires_grad": [False],
                    "shape": [(1, 100352, 1), (1, 81920, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_invstd"],
                    "requires_grad": [False],
                    "shape": [(1, 100352, 1), (1, 81920, 1)],
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
            other=[4.13517, 4.13517, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(159, 2), (34, 2), (1, 1, 1, 1), (1, 1024, 1, 1)],
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
                    "shape": [(224, 144, 384), (1, 50176, 192)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(384, 384), (768, 192)],
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

    'log': dict(
        name=["log"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(41,), (122,)],
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
                    "shape": [(86, 2), (36, 2), (1, 1024, 1, 1), (1, 1, 1, 1)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "shape": [(86, 2), (36, 2), (1, 1024, 1, 1), (1, 1, 1, 1)],
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
                    "shape": [(1, 256, 24, 18), (1, 256, 224, 160), (113, 2), (91, 2), (117,), (253,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 256, 24, 18), (1, 256, 224, 160), (113, 2), (91, 2), (1,), (1,)],
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
                    "shape": [(420,), (560,)],
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
            value=[0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(199,), (330,), (1, 256, 160, 176), (1, 256, 18, 10), (110, 2), (210, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "requires_grad": [False],
                    "shape": [(199,), (330,), (1, 256, 160, 176), (1, 256, 18, 10), (110, 2), (210, 2)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'masked_fill_case_2': dict(
        name=["masked_fill"],
        interface=["torch"],
        para=dict(
            value=[-100, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(48, 144, 144), (640, 144, 144)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "requires_grad": [False],
                    "shape": [(48, 144, 144), (640, 144, 144)],
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
                    "shape": [(27280, 6), (42966, 13)],
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
                    "shape": [(68, 4), (3, 4), (22, 4), (7, 4), (43, 4), (15, 4), (2, 4), (1, 4), (12, 4), (11, 4), (23, 4), (5, 4), (4, 4), (6, 4), (8, 4), (18, 4)],
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
                    "requires_grad": [True],
                    "shape": [(98, 2), (38192, 9), (1, 256, 12, 18), (1, 256, 48, 128), (14322, 1, 2), (44330, 1, 2), (66,), (181,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [True],
                    "shape": [(98, 2), (1,), (1, 256, 12, 18), (1, 256, 48, 128), (1, 28, 2), (1, 10, 2), (1,), (1,)],
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
            dim=[None, None, None, None, [0], [0], None, None],
            keepdim=[False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (2304,), (384,), (45, 45), (45, 34), (256, 1536, 1, 1), (256, 256, 3, 3)],
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
                    "shape": [(82, 2), (130, 2), (45, 4, 9), (45, 4, 21)],
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
                    "requires_grad": [True],
                    "shape": [(46035, 1, 2), (5456, 1, 2), (206, 2), (8, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 17, 2), (1, 3, 2), (206, 2), (8, 2)],
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
                    "shape": [(1, 4, 48, 36), (1, 256, 128, 112), (1,), (60,), (27621, 4), (8320, 80), (1, 110592, 192), (1, 20480, 192)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(), (1, 256, 1, 1), (1,), (60,), (27621, 4), (8320, 1), (1, 1, 1), (1, 1, 1)],
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
                    "shape": [(146,), (94,), (112, 80), (252, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(146,), (94,), (112, 80), (252, 80)],
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
            other=[0.999997, 0.999, 0.999999, 0.999999, 0.999999, 0.9, 0.999999, 0.999998],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(27, 256, 3, 3), (64, 256, 1, 1), (1536, 1536), (384, 384), (1,), (1536,), (), ()],
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
            other=[0.5, 0.5, 0.176777, 0.176777, 64, 128, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(192, 2), (8, 2), (576, 6, 144, 32), (154, 12, 144, 32), (30,), (10,), (), ()],
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
            other=[-100, -100, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(161,), (156,), (480, 144, 144), (25, 144, 144)],
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
                    "shape": [(215, 2), (154, 2), (8,), (118,)],
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
                    "shape": [(12800,), (1920,), (20480, 80), (25600, 80)],
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
                    "shape": [(1, 256, 1, 1), (80, 256, 1, 1), (4, 256, 1, 1)],
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
                    "shape": [(18414, 4, 2), (34100, 19, 2)],
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
                    "shape": [(1, 1, 1, 1), (1, 1, 1, 1), (1, 64, 1, 1), (1, 64, 1, 1)],
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
            repeats=[(1, 1), (1, 72), (128,), (24,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(23, 4), (56, 1), (56,), (18,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'roll': dict(
        name=["roll"],
        interface=["torch"],
        para=dict(
            shifts=[(6, 6), (6, 6)],
            dims=[(2, 1), (1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 168, 132, 192), (1, 72, 72, 384)],
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
                    "requires_grad": [True],
                    "shape": [(1, 9, 20, 22), (1, 9, 16, 30), (195,), (30720,), (440, 80), (64, 80)],
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
            dim=[-1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(760, 6, 144, 144), (770, 6, 144, 144)],
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
            dim=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            descending=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(23,), (0,), (4,), (2,), (68,), (5,), (6,), (7,), (8,), (43,), (15,), (22,), (3,), (18,), (1,), (11,), (12,)],
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
                    "shape": [(44330, 46), (36828, 18), (256, 256, 3, 3), (1024, 64, 1, 1), (108,), (16,), ()],
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
            dim=[1, 1, 0, 0, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(560,), (560,), (560,), (560,)], [(392,), (392,), (392,), (392,)], [(33418,)], [(38192, 4)], [(180,), (180,)], [(42966,), (42966,)]],
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
                    "shape": [(45, 40), (45, 54)],
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
                    "shape": [(5, 2), (5456, 1), (3, 506, 757), (3, 1616, 1078), (41,), (50,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(5, 2), (5456, 1), (3, 1, 1), (3, 1, 1), (41,), (50,)],
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
                    "shape": [(196,), (4,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(196,), (4,)],
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
            other=[1, 1, 0.5, 0.5],
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(88,), (273,), (1, 1024, 1, 1), (1, 1024, 1, 1)],
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
            dim=[[-1], [-1], [0], [1, 2, 3], None, None, None, None, None],
            keepdim=[False, False, True, True, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(40920, 8, 2), (27280, 18, 2), (946, 6, 144, 144), (1, 256, 64, 128), (3136,), (60,), (12288, 4), (3200, 80), ()],
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
            k=[9, 9],
            dim=[0, 0],
            largest=[False, False],
            sorted=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(396, 4), (5120, 20)],
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
            start=[-0.0765466, -0.0968246, -0.0360844, 0, -0.0578638],
            end=[0.0765466, 0.0968246, 0.0360844, 1, 0.0578638],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 768, 1, 1), (256, 384, 1, 1), (256, 256, 3, 3), (1, 1, 1), (256, 1536, 1, 1)],
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
                    "shape": [(45973,), (314,)],
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
            mode=['linear', 'linear'],
            size=[(20, 22), (18, 26)],
            align_corners=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 256, 10, 11), (1, 256, 9, 13)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'interpolate_case_2': dict(
        name=["interpolate"],
        interface=["torch.nn.functional"],
        para=dict(
            mode=['nearest', 'nearest'],
            size=[(144, 144), (104, 64)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 256, 72, 72), (1, 256, 52, 32)],
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
                    "shape": [(109,), (198,), (58, 2), (110, 2), (1, 256, 18, 18), (1, 256, 9, 14)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(109,), (198,), (58, 2), (110, 2), (1, 256, 18, 18), (1, 256, 9, 14)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(109,), (198,), (), (), (1, 256, 18, 18), (1, 256, 9, 14)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
