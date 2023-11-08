import numpy as np

yolov5_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1, 0.0005, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(161,), (464,), (237, 2), (78, 2), (64, 64, 3, 3), (1, 32, 160, 160), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(161,), (464,), (237, 2), (78, 2), (64, 64, 3, 3), (1, 32, 160, 160), ()],
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
            alpha=[0.800059, 1, -1.69079e-06, -3.09978e-06, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (154,), (255, 512, 1, 1), (128, 256, 1, 1), (36, 2), (247, 80), (1, 3, 85, 80, 80), (1, 3, 85, 40, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64,), (154,), (255, 512, 1, 1), (128, 256, 1, 1), (36, 2), (247, 80), (1, 3, 85, 80, 80), (1, 3, 85, 40, 40)],
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
            other=[-5.29832, -6.68461, -3.91202, -4.88027, 1],
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3,), (3,), (3,), (3, 80), ()],
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
            other=[1e-07, 1e-07, 1, 1, 0],
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(11,), (213,), (1, 32, 160, 160), (1, 64, 160, 160), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'atan': dict(
        name=["atan"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(94,), (406,)],
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
            momentum=[0.03, 0.03],
            eps=[0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 64, 160, 160), (1, 64, 80, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(64,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(64,), (64,)],
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
                    "shape": [(62, 80), (306, 80), (2, 3, 20, 20), (1, 3, 20, 20)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(62, 80), (306, 80), (2, 3, 20, 20), (1, 3, 20, 20)],
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
            dim=[1, 1, 1, 1, 0, 0, 1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(118, 2), (118, 2)], [(201, 2), (201, 2)], [(21, 1), (21, 1), (21, 4)], [(1, 1), (1, 1), (1, 4)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(3456,), (32,), (32,), (18432,), (64,), (64,), (2048,), (32,), (32,), (2048,), (32,), (32,), (4096,), (64,), (64,), (1024,), (32,), (32,), (9216,), (32,), (32,), (73728,), (128,), (128,), (8192,), (64,), (64,), (8192,), (64,), (64,), (16384,), (128,), (128,), (4096,), (64,), (64,), (36864,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (294912,), (256,), (256,), (32768,), (128,), (128,), (32768,), (128,), (128,), (65536,), (256,), (256,), (16384,), (128,), (128,), (147456,), (128,), (128,), (16384,), (128,), (128,), (147456,), (128,), (128,), (16384,), (128,), (128,), (147456,), (128,), (128,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (131072,), (256,), (256,), (262144,), (512,), (512,), (65536,), (256,), (256,), (589824,), (256,), (256,), (131072,), (256,), (256,), (524288,), (512,), (512,), (131072,), (256,), (256,), (65536,), (128,), (128,), (65536,), (128,), (128,), (65536,), (256,), (256,), (16384,), (128,), (128,), (147456,), (128,), (128,), (32768,), (128,), (128,), (16384,), (64,), (64,), (16384,), (64,), (64,), (16384,), (128,), (128,), (4096,), (64,), (64,), (36864,), (64,), (64,), (147456,), (128,), (128,), (589824,), (256,), (256,), (32768,), (128,), (128,), (32768,), (128,), (128,), (65536,), (256,), (256,), (16384,), (128,), (128,), (147456,), (128,), (128,), (131072,), (256,), (256,), (131072,), (256,), (256,), (262144,), (512,), (512,), (65536,), (256,), (256,), (589824,), (256,), (256,), (32640,), (255,), (65280,), (255,), (130560,), (255,), (3,), (3,), (32,), (32,), (64,), (64,), (32,), (32,), (32,), (32,), (64,), (64,), (32,), (32,), (32,), (32,), (128,), (128,), (64,), (64,), (64,), (64,), (128,), (128,), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (256,), (256,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (512,), (512,), (256,), (256,), (256,), (256,), (256,), (256,), (512,), (512,), (256,), (256,), (128,), (128,), (128,), (128,), (256,), (256,), (128,), (128,), (128,), (128,), (128,), (128,), (64,), (64,), (64,), (64,), (128,), (128,), (64,), (64,), (64,), (64,), (128,), (128,), (256,), (256,), (128,), (128,), (128,), (128,), (256,), (256,), (128,), (128,), (128,), (128,), (256,), (256,), (256,), (256,), (512,), (512,), (256,), (256,), (256,), (256,), (18,), (10,), (3,)], [(2, 256, 20, 20), (2, 256, 20, 20), (2, 256, 20, 20), (2, 256, 20, 20)], [(1, 256, 20, 20), (1, 256, 20, 20), (1, 256, 20, 20), (1, 256, 20, 20)], [(19, 6)]],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'clamp_min': dict(
        name=["clamp_min"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(255, 80), (135, 80), (2, 3, 40, 40), (1, 3, 80, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
            min=[0, 0, 0, 0],
            max=[None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(165,), (20,), (224, 2), (196, 2)],
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
            padding=[(1, 1), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 64, 80, 80), (1, 256, 40, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 64, 3, 3), (256, 256, 1, 1)],
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
                    "shape": [(304,), (189,), (3, 18, 2), (3, 56, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(304,), (189,), (3, 1, 2), (3, 1, 2)],
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
            other=[640, 640],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(40, 2), (21, 2)],
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
            other=[2, 2, 64, 4, 1, 1, 2400, 9600],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(154, 2), (168, 2), (64,), (60,), (), (), (2, 3, 20, 20), (2, 3, 40, 40)],
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
                    "shape": [(152, 2), (11, 2), (209,), (77,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(152, 2), (11, 2), (209,), (77,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'exp': dict(
        name=["exp"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(137, 80), (204, 80), (1, 3, 40, 40), (2, 3, 20, 20)],
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
            value=[1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(58,), (180,), (366, 80), (69, 80), (2, 3, 4, 80, 80), (2, 3, 85, 80, 80), (1, 32, 320, 320), (2, 256, 40, 40), (), ()],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0, -1, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(164, 2), (76, 2), (253,), (78,), (2, 3, 20, 20), (1, 3, 40, 40)],
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
                    "shape": [(198, 2), (156, 2), (287,), (153,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(198, 2), (156, 2), (287,), (153,)],
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
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(5, 2), (13, 2)],
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
                    "shape": [(2, 3, 80, 80, 80), (2, 3, 4, 40, 40), (5, 36, 7), (3, 12, 7), (3, 2), (3, 2), (7,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(2, 3, 80), (2, 3, 4, 40, 40), (5, 36), (3, 12), (3,), (3, 2), (7,)],
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
            accumulate=[False, False, True, True, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 3, 40, 40), (1, 3, 40, 40), (2, 3, 80, 20, 20), (1, 3, 80, 40, 40), (151, 80), (164, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(94,), (136,), (30,), (136,), (151,), (164,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-1, high=1),
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(94, 3, 40, 40), (136, 3, 40, 40), (30, 3, 80, 20, 20), (136, 3, 80, 40, 40), (), ()],
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
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(101,), (15,)],
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
            weight=[0.981181, 0.962235, 0.954092, 0.982654],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(128, 128, 3, 3), (128, 512, 1, 1), (255,), (32,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["end"],
                    "requires_grad": [False],
                    "shape": [(128, 128, 3, 3), (128, 512, 1, 1), (255,), (32,)],
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
                    "shape": [(15, 80), (321, 80), (1, 3, 40, 40), (2, 3, 80, 80)],
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
                    "shape": [(303,), (93,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["other"],
                    "shape": [(303,), (93,)],
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
                    "shape": [(241, 2), (266, 2), (89,), (138,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(241, 2), (266, 2), (89,), (138,)],
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
            other=[4, 4],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 18), (3, 60)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(158,), (147,), (82, 2), (136, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "requires_grad": [False],
                    "shape": [(158,), (147,), (82, 2), (136, 2)],
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
            dim=[2, 2],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(3, 56, 2), (3, 42, 2)],
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
            kernel_size=[(5, 5), (5, 5)],
            stride=[(1, 1), (1, 1)],
            padding=[(2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1)],
            ceil_mode=[False, False],
            return_indices=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 20, 20), (1, 256, 20, 20)],
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
                    "shape": [(309,), (232,), (342, 2), (109, 2), (3, 27, 2), (3, 34, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(309,), (232,), (342, 2), (109, 2), (3, 27, 2), (3, 34, 2)],
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
            dim=[None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(174,), (245,), (60, 80), (224, 80), (255, 512, 1, 1), (64, 32, 3, 3)],
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
                    "shape": [(169,), (207,), (118, 2), (48, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(169,), (207,), (118, 2), (48, 2)],
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
                    "shape": [(192, 2), (324, 2), (156,), (147,), (1, 256, 40, 40), (2, 32, 320, 320), (3, 16, 7), (3, 45, 7)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(192, 2), (324, 2), (156,), (147,), (1, 256, 40, 40), (2, 32, 320, 320), (7,), (7,)],
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
                    "shape": [(78, 80), (15, 80), (2, 3, 80, 80), (1, 3, 80, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(78, 80), (15, 80), (2, 3, 80, 80), (1, 3, 80, 80)],
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
            other=[0.800049, 0.800079, 0.800053, 0.845649],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 32, 3, 3), (32, 32, 3, 3), (255,), (512,)],
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
            other=[0.5, 2, 2, 0.405285, 1, 1, 1, 1, 1, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(218, 2), (253, 2), (97,), (105,), (3, 22, 2), (3, 56, 2), (255, 256, 1, 1), (255, 512, 1, 1), (), ()],
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
            other=[-100, -100, -100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(140, 80), (375, 80), (2, 3, 80, 80), (1, 3, 40, 40)],
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
                    "shape": [(42,), (140,), (97, 80), (150, 80), (1, 3, 20, 20), (2, 3, 20, 20)],
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
                    "shape": [(5, 94), (3, 34)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
            ],
        ),
    ),

    'pow': dict(
        name=["pow"],
        interface=["torch"],
        para=dict(
            exponent=[2, 1, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(369, 2), (329, 2), (216,), (118,)],
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
                    "shape": [(3, 60, 2), (3, 40, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'remainder': dict(
        name=["remainder"],
        interface=["torch"],
        para=dict(
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(33, 2), (60, 2)],
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
            repeats=[(1, 3, 1), (1, 45, 1), (5, 1, 1), (5, 1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(5, 1, 2), (5, 1, 2), (42, 7), (36, 7)],
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
                    "shape": [(287, 4), (232, 4), (1, 64, 80, 80), (1, 128, 40, 40)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'silu': dict(
        name=["silu"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 32, 320, 320), (1, 128, 40, 40)],
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
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(89,), (89,), (89,), (89,), (89,)], [(20,), (20,), (20,), (20,), (20,)], [(3, 640, 640), (3, 640, 640)], [(3, 640, 640)]],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(196,), (48,), (48, 2), (272, 2), (2, 3, 20, 20), (1, 3, 20, 20), (3, 640, 640)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(196,), (48,), (48, 2), (272, 2), (2, 3, 20, 20), (1, 3, 20, 20), (3, 1, 1)],
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
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(150, 80), (122, 80), (2, 3, 40, 40), (2, 64, 80, 80)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(150, 80), (122, 80), (2, 3, 40, 40), (2, 64, 80, 80)],
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
            other=[0.5, 0.5],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(222, 2), (216, 2)],
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
            dim=[None, None, None],
            keepdim=[False, False, False],
            dtype=[None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (2, 240, 80, 80), (2, 12, 80, 80)],
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
            start=[-0.0589256, -0.0441942],
            end=[0.0589256, 0.0441942],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 32, 3, 3), (128, 512, 1, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        interface=["torch.nn.functional"],
        para=dict(
            mode=['nearest', 'nearest', 'nearest', 'nearest'],
            size=[(40, 40), (80, 80), (80, 80), (40, 40)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 20, 20), (2, 128, 40, 40), (1, 128, 40, 40), (1, 256, 20, 20)],
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
                    "shape": [(115,), (253,), (51, 2), (86, 2)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(115,), (253,), (51, 2), (86, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(), (), (51, 2), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
