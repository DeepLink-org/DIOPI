import numpy as np

unet_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[0.0005, 0.0005, 0.0005, 0.0005, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 3, 3), (128, 64, 3, 3), (512,), (1024,), ()],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 3, 3, 3), (128, 64, 3, 3), (512,), (1024,), ()],
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
            alpha=[-0.00994652, -0.00994452, -0.00994613, -0.00994597],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 512, 3, 3), (512, 1024, 3, 3), (64,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1024, 512, 3, 3), (512, 1024, 3, 3), (64,), (64,)],
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
            other=[0, 1.19209e-07],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (1,)],
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
            training=[True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 64, 512, 1024), (4, 1024, 32, 64), (4, 512, 64, 128), (4, 256, 128, 256), (4, 64, 256, 512), (4, 128, 256, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (1024,), (512,), (256,), (64,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (1024,), (512,), (256,), (64,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(64,), (1024,), (512,), (256,), (64,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(64,), (1024,), (512,), (256,), (64,), (128,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=["bernoulli"],
        interface=["torch.nn.functional"],
        para=dict(
            p=[0.9],
            generator=[None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(4, 64, 1, 1)],
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
            dim=[0, 1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(4, 128, 256, 512), (4, 128, 256, 512)], [(4, 256, 128, 256), (4, 256, 128, 256)], [(1728,), (64,), (64,), (36864,), (64,), (64,), (73728,), (128,), (128,), (147456,), (128,), (128,), (294912,), (256,), (256,), (589824,), (256,), (256,), (1179648,), (512,), (512,), (2359296,), (512,), (512,), (4718592,), (1024,), (1024,), (9437184,), (1024,), (1024,), (73728,), (64,), (64,), (36864,), (64,), (64,), (8192,), (64,), (64,), (294912,), (128,), (128,), (147456,), (128,), (128,), (32768,), (128,), (128,), (1179648,), (256,), (256,), (589824,), (256,), (256,), (131072,), (256,), (256,), (4718592,), (512,), (512,), (2359296,), (512,), (512,), (524288,), (512,), (512,), (1216,), (19,), (36864,), (64,), (64,), (1216,), (19,), (73728,), (64,), (64,), (3,), (3,), (64,), (64,), (64,), (64,), (128,), (128,), (128,), (128,), (256,), (256,), (256,), (256,), (512,), (512,), (512,), (512,), (1024,), (1024,), (1024,), (1024,), (64,), (64,), (64,), (64,), (64,), (64,), (128,), (128,), (128,), (128,), (128,), (128,), (256,), (256,), (256,), (256,), (256,), (256,), (512,), (512,), (512,), (512,), (512,), (512,), (64,), (64,), (64,), (64,)]],
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
            padding=[(1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 64, 512, 1024), (4, 128, 256, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 64, 3, 3), (128, 128, 3, 3)],
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
                    "shape": [(3, 512, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
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
            other=[0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 64, 1, 1)],
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
            other=[1, 1, 1, 2097152],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (4, 512, 1024)],
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
                    "shape": [(1, 4, 512, 1024)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4, 512, 1024)],
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
            value=[0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(19,), ()],
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
                    "shape": [(4, 512, 1024), (4, 512, 1024), (1, 4, 512, 1024), (1, 4, 512, 1024)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
                },
                {
                    "ins": ["indices"],
                    "requires_grad": [False],
                    "shape": [(4, 512, 1024), (4, 512, 1024), (4, 512, 1024), (4, 512, 1024)],
                    "dtype": [np.bool_],
                    "gen_fn": "Genfunc.mask",
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
                    "shape": [(4, 19, 512, 1024)],
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
            kernel_size=[(2, 2), (2, 2), (2, 2), (2, 2)],
            stride=[(2, 2), (2, 2), (2, 2), (2, 2)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            ceil_mode=[False, False, False, False],
            return_indices=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 256, 128, 256), (4, 128, 256, 512), (4, 512, 64, 128), (4, 64, 512, 1024)],
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
                    "shape": [(1,), (1024,), (64, 64, 3, 3), (64, 3, 3, 3), (4, 512, 1024), ()],
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
                    "shape": [(4, 64, 512, 1024), (4, 64, 512, 1024), (4, 64, 256, 512), (4, 64, 256, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(4, 64, 1, 1), (4, 64, 1, 1), (4, 64, 1, 1), (4, 64, 1, 1)],
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
            other=[4.81967e-05, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (1024,), (512, 512, 3, 3), (512, 1024, 1, 1)],
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
            other=[1, 1, 1, 1, 0.4, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 64, 3, 3), (19, 64, 1, 1), (512,), (64,), (), ()],
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
            other=[255],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 512, 1024)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(4, 512, 1024)],
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
            mean=[0, 0],
            std=[0.0589256, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 64, 3, 3), (19, 64, 1, 1)],
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
                    "shape": [(4, 64, 512, 1024), (4, 256, 128, 256), (4, 1024, 32, 64), (4, 512, 64, 128), (4, 64, 256, 512), (4, 128, 256, 512)],
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
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(3, 512, 1024), (3, 512, 1024), (3, 512, 1024), (3, 512, 1024)], [(1, 512, 1024), (1, 512, 1024), (1, 512, 1024), (1, 512, 1024)]],
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
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 512, 1024)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
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
            dim=[[0], [0]],
            keepdim=[True, True],
            dtype=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2055812,), (1909302,)],
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
            k=[1],
            dim=[1],
            largest=[True],
            sorted=[True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 19, 512, 1024)],
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
            mode=['linear', 'linear', 'linear', 'linear', 'linear', 'linear'],
            size=[(512, 1024), (512, 1024), (256, 512), (64, 128), (128, 256), (512, 1024)],
            align_corners=[0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(4, 128, 256, 512), (4, 19, 512, 1024), (4, 256, 128, 256), (4, 1024, 32, 64), (4, 512, 64, 128), (4, 19, 256, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    # 'nll_loss2d': dict(
    #     name=["nll_loss2d"],
    #     interface=["torch.nn.functional"],
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ["input"],
    #                 "requires_grad": [True],
    #                 "shape": [(4, 19, 512, 1024)],
    #                 "dtype": [np.float32],
    #                 "gen_fn": "Genfunc.randn",
    #             },
    #             {
    #                 "ins": ["grad_output"],
    #                 "requires_grad": [False],
    #                 "shape": [(4, 512, 1024)],
    #                 "dtype": [np.float32],
    #                 "gen_fn": "Genfunc.randn",
    #             },
    #         ],
    #     ),
    # ),

}
