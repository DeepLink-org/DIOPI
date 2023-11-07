import numpy as np

detr_config = {
    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 4)],
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
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 561, 256), (2, 819, 256), (43, 1), (38, 1), (), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 561, 256), (2, 819, 256), (43, 1), (38, 1), (), (200,)],
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
            alpha=[1, 1, 1, 1, 0.1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 512, 93, 68), (2, 1024, 42, 40), (2, 693, 256), (2, 512, 256), (81,), (200,), (200, 4), (200, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 512, 93, 68), (2, 1024, 42, 40), (2, 693, 256), (2, 512, 256), (81,), (200,), (200, 4), (200, 2)],
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
            other=[1, 1, 1e-06, 1e-06, 1e-08, 1e-08, 1e-08, 1e-08, 1e-06, 0],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(48,), (22,), (2, 17, 1), (2, 1, 17), (128, 128, 3, 3), (512, 128, 1, 1), (100, 256), (256, 256), (), ()],
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
            value=[-1.10923e-05, -1.00008e-05, -0.000100305, -0.000100086, -0.000125927, -0.000103967],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(256, 512, 1, 1), (512, 2048, 1, 1), (256,), (256,), (81, 256), (256, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(256, 512, 1, 1), (512, 2048, 1, 1), (256,), (256,), (81, 256), (256, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(256, 512, 1, 1), (512, 2048, 1, 1), (256,), (256,), (81, 256), (256, 256)],
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
                    "shape": [(512, 2048, 1, 1), (512, 256, 1, 1), (100, 256), (256, 256), (256,), (4,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(512, 2048, 1, 1), (512, 256, 1, 1), (100, 256), (256, 256), (256,), (4,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(512, 2048, 1, 1), (512, 256, 1, 1), (100, 256), (256, 256), (256,), (4,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(200, 4)],
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
            start=[0],
            end=[128],
            step=[1],
        ),
    ),

    'baddbmm': dict(
        name=["baddbmm"],
        interface=["torch"],
        para=dict(
            beta=[1, 1],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(16, 1, 475), (16, 1, 775)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["batch1"],
                    # "requires_grad": [True],
                    "shape": [(16, 475, 32), (16, 100, 32)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["batch2"],
                    # "requires_grad": [True],
                    "shape": [(16, 32, 475), (16, 32, 775)],
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
                    "shape": [(2, 128, 171, 163), (2, 256, 248, 240)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [False],
                    "shape": [(128,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [False],
                    "shape": [(128,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(128,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(128,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
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
                    # "requires_grad": [True],
                    "shape": [(16, 285, 285), (16, 32, 792)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mat2"],
                    "requires_grad": [True],
                    "shape": [(16, 285, 32), (16, 792, 792)],
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
            dim=[3, 3, 1, 1, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(2, 34, 28, 128), (2, 34, 28, 128)], [(2, 34, 20, 128), (2, 34, 20, 128)], [(51, 1), (51, 1), (51, 1), (51, 1)], [(58, 1), (58, 1), (58, 1), (58, 1)], [(256,), (256,), (256,)], [(256, 256), (256, 256), (256, 256)], [(9408,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (524288,), (256,), (20736,), (81,), (65536,), (256,), (65536,), (256,), (1024,), (4,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (196608,), (768,), (65536,), (256,), (196608,), (768,), (65536,), (256,), (524288,), (2048,), (524288,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (25600,), (3,), (3,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'cdist': dict(
        name=["cdist"],
        interface=["torch"],
        saved_args=dict(output=0),
        para=dict(
            p=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            compute_mode=[None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["x1"],
                    "requires_grad": [True],
                    "shape": [(100, 4), (100, 4), (100, 4), (100, 4), (100, 4), (100, 4), (100, 4), (100, 4), (100, 4), (100, 4)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["x2"],
                    "requires_grad": [False],
                    "shape": [(2, 4), (7, 4), (31, 4), (54, 4), (12, 4), (43, 4), (19, 4), (9, 4), (48, 4), (45, 4)],
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
            min=[0, 0, 1, 0, None],
            max=[None, None, None, None, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(100, 12, 2), (100, 45, 2), (1,), (200, 2), ()],
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
            padding=[(3, 3), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 3, 663, 819), (2, 64, 197, 237)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [False],
                    "shape": [(64, 3, 7, 7), (256, 64, 1, 1)],
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

    'cos': dict(
        name=["cos"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 42, 21, 64), (2, 20, 34, 64)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'cumsum': dict(
        name=["cumsum"],
        interface=["torch"],
        para=dict(
            dim=[2, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 20, 39), (2, 31, 37)],
                    "dtype": [np.int32],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(2, 38, 26, 1), (2, 25, 25, 1), (3, 480, 607), (3, 970, 608), (100, 29), (100, 42), (200,), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(128,), (128,), (3, 1, 1), (3, 1, 1), (100, 29), (100, 42), (200,), (200,)],
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
            other=[0.994624, 0.945673, 0.989683, 0.998222, 0.609927, 0.812138],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 256, 1, 1), (512, 2048, 1, 1), (100, 256), (256, 2048), (4,), (2048,)],
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
            other=[5.65685, 5.65685, 45, 49.7, 2, 2, 2, 128],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 680, 32), (16, 1188, 32), (), (), (20, 1), (6, 1), (200,), (128,)],
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
            p=[0.1, 0.1],
            training=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 884, 256), (2, 1312, 2048)],
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
                    "shape": [(200, 2), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(200, 2), (1,)],
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
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100,)],
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
            value=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 648, 909), (2, 796, 731), (2, 405), (2, 663), (256, 1024, 1, 1), (512, 128, 1, 1), (100,), (256,), (), ()],
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
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 2)],
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
                    "shape": [(200, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(200, 2)],
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
            other=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 4), (100,)],
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
                    "shape": [(3, 910, 704), (3, 736, 1122), (44, 4), (42, 4), (55,), (100,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(3,), (3,), (44,), (42,), (55,), (0,)],
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
                    "shape": [(100, 4), (100, 4), (100,), (100,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["indices1"],
                    "requires_grad": [False],
                    "shape": [(40,), (46,), (59,), (23,)],
                    "dtype": [np.int64],
                    "gen_fn": dict(fn="Genfunc.randint", low=-100, high=100),
                },
                {
                    "ins": ["values"],
                    "requires_grad": [False],
                    "shape": [(40, 4), (), (59,), (23,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        interface=["torch.nn.functional"],
        para=dict(
            normalized_shape=[(256,), (256,)],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 870, 256), (2, 902, 256)],
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
                {
                    "ins": ["save_mean"],
                    "requires_grad": [False],
                    "shape": [(2, 870, 1), (2, 902, 1)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["save_invstd"],
                    "requires_grad": [False],
                    "shape": [(2, 870, 1), (2, 902, 1)],
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
                    "shape": [(2048, 256), (2660, 256), (2, 1020, 2048), (980, 2, 256), (6, 2, 100, 256), (6, 2, 100, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 256), (256, 256), (256, 2048), (256, 256), (256, 256), (4, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (4,)],
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
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(200, 81)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
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
                    "shape": [(200, 2), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(200, 2), (1,)],
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
            value=['-inf', '-inf', 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 1184), (2, 1254), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["mask"],
                    "requires_grad": [False],
                    "shape": [(2, 1184), (2, 1254), (200,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(2, 64, 393, 342), (2, 64, 438, 504)],
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
                    "shape": [(100, 49), (100, 28), (100, 1, 2), (100, 1, 2), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1,), (1,), (1, 31, 2), (1, 2, 2), (1,)],
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
            dim=[[1], [1], None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 8, 704, 704), (2, 8, 100, 578), (256, 256), (2048, 256), (1024,), (128,), ()],
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
                    "shape": [(100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (100, 1, 2), (200, 2)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(1, 2, 2), (1, 19, 2), (1, 63, 2), (1, 28, 2), (1, 66, 2), (1, 33, 2), (1, 49, 2), (1, 48, 2), (1, 21, 2), (1, 45, 2), (200, 2)],
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
                    "shape": [(100, 58), (100, 70), (2, 544, 256), (16, 100, 1085), (6,), (48,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(100, 58), (100, 70), (2, 544, 256), (16, 100, 1085), (6,), (48,)],
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
                    "shape": [(512, 1024, 1, 1), (512, 512, 3, 3), (768, 256), (2048, 256), (81,), (4,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), ()],
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
            other=[0.999, 0.9, 0.9, 1, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 256), (2048, 256), (256,), (256,), (256, 1024, 1, 1), (1024, 512, 1, 1)],
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
            other=[6.28319, 1.11111, 5, 1, 1, 1, 1, 5, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 21, 40), (16, 100, 315), (100, 48), (100, 66), (512, 2048, 1, 1), (512, 128, 1, 1), (), (), (256,), (81,)],
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
            reduction=['none'],
            ignore_index=[-100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(200, 81)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["target"],
                    "requires_grad": [False],
                    "shape": [(200,)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [False],
                    "shape": [(81,)],
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
                    "shape": [(100, 61), (100, 24), (200,)],
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
                    "shape": [(100,)],
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
            p=[2, 2, 2, 2, 2, 2],
            dim=[(0, 1, 2, 3), (0, 1, 2, 3), (0, 1), (0, 1), (0,), (0,)],
            keepdim=[False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(1024, 256, 1, 1), (512, 256, 1, 1), (2048, 256), (256, 256), (4,), (2048,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'pow_tensor': dict(
        name=["pow_tensor"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["exponent"],
                    "requires_grad": [False],
                    "shape": [(128,)],
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
                    "shape": [(6, 2, 100, 256)],
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
                    "shape": [(2, 64, 272, 331), (2, 1024, 36, 65), (2, 1188, 2048), (2, 925, 2048)],
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
            repeats=[(2, 1, 1), (100, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "requires_grad": [True],
                    "shape": [(1, 100, 256), (1, 4)],
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
                    "shape": [(200, 4)],
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
                    "shape": [(6, 2, 100, 4)],
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
                    "requires_grad": [False],
                    "shape": [(2, 27, 34, 64), (2, 25, 24, 64)],
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
            dim=[-1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 459, 459), (16, 899, 899), (100, 81)],
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
                    "shape": [(81, 256), (256, 256), (2048, 1024, 1, 1), (512, 1024, 1, 1), (4,), (2048,)],
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
            dim=[0, 0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(3, 570, 683), (3, 570, 683)], [(3, 644, 959), (3, 644, 959)], [(100, 30), (100, 30), (100, 30)], [(100, 55), (100, 55), (100, 55)], [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()], [(2, 100, 81), (2, 100, 81), (2, 100, 81), (2, 100, 81), (2, 100, 81), (2, 100, 81)], [(2, 100, 256), (2, 100, 256), (2, 100, 256), (2, 100, 256), (2, 100, 256), (2, 100, 256)]],
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
                    "shape": [(3, 608, 947), (3, 640, 700), (32, 1), (29, 1), (20,), (70,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (32, 1), (29, 1), (20,), (70,)],
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
            other=[1, 1],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(63,), (70,)],
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
            dim=[[0], [0], None, [0], None],
            keepdim=[False, False, False, False, False],
            dtype=[None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 100, 33), (3, 100, 32), (200,), (2, 1, 100, 256), (200, 4)],
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
            start=[-0.051031, -0.108253, -0.051031, -0.0765466, -0.051031],
            end=[0.051031, 0.108253, 0.051031, 0.0765466, 0.051031],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 256), (256, 256), (256, 2048), (768, 256), (256, 2048, 1, 1)],
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
                    "shape": [(23,), (57,)],
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
            size=[(29, 23), (23, 22)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(2, 1, 916, 708), (2, 1, 722, 704)],
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
                    "shape": [(200, 2), (200, 2), (200,)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(200, 2), (200, 2), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "requires_grad": [False],
                    "shape": [(), (200, 2), (200,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
