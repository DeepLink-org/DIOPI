import numpy as np

crnn_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 256)],
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
            alpha=[-1, -1, 1, 1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (512,), (64, 1024), (1024, 512), (128, 64, 3, 3), (512, 512, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64,), (512,), (64, 1024), (1024, 512), (128, 64, 3, 3), (512, 512, 3, 3)],
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
            other=[1e-06, 1e-06, 0, 1e-06, 1e-06, 1e-06, 1e-06],
            alpha=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 64, 3, 3), (512, 256, 3, 3), (), (37,), (64,), (1024, 512), (256, 512)],
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
            value=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(256, 256, 3, 3), (512, 256, 3, 3), (1024, 512), (1024, 256), (37,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [False],
                    "shape": [(256, 256, 3, 3), (512, 256, 3, 3), (1024, 512), (1024, 256), (37,), (64,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [False],
                    "shape": [(256, 256, 3, 3), (512, 256, 3, 3), (1024, 512), (1024, 256), (37,), (64,)],
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
            training=[True, True, True],
            momentum=[0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 256, 8, 25), (64, 512, 4, 26), (64, 512, 1, 26)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256,), (512,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (512,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad": [False],
                    "shape": [(256,), (512,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["running_var"],
                    "requires_grad": [False],
                    "shape": [(256,), (512,), (512,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.positive",
                },
            ],
        ),
    ),

    # FIXME ctc_loss的input_lengths参数输入tuple报错
    # 'ctc_loss': dict(
    #     name=["ctc_loss"],
    #     interface=["CustomizedTest"],
    #     para=dict(
    #         input_lengths=[(26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26), (26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26)],
    #         target_lengths=[(8, 8, 9, 12, 10, 8, 12, 6, 9, 4, 8, 9, 10, 10, 7, 9, 8, 8, 10, 8, 8, 6, 5, 9, 5, 7, 5, 8, 8, 8, 13, 9, 9, 10, 8, 7, 6, 7, 6, 7, 8, 8, 7, 6, 10, 9, 8, 12, 12, 9, 7, 5, 7, 9, 6, 10, 5, 6, 7, 9, 14, 10, 10, 13), (4, 13, 4, 6, 8, 8, 8, 8, 9, 9, 7, 6, 10, 13, 4, 7, 10, 7, 10, 10, 7, 12, 7, 8, 9, 10, 6, 4, 6, 5, 8, 12, 5, 8, 10, 12, 11, 11, 11, 9, 7, 4, 8, 6, 10, 11, 11, 7, 7, 3, 8, 14, 6, 14, 15, 11, 7, 8, 4, 8, 9, 6, 4, 4)],
    #         blank=[36, 36],
    #         zero_infinity=[False, False],
    #     ),
    #     tensor_para=dict(
    #         args=[
    #             {
    #                 "ins": ["log_probs"],
    #                 "requires_grad": [True],
    #                 "shape": [(26, 64, 37), (26, 64, 37)],
    #                 "dtype": [np.float32],
    #                 "gen_fn": "Genfunc.randn",
    #             },
    #             {
    #                 "ins": ["targets"],
    #                 "requires_grad": [False],
    #                 "shape": [(531,), (524,)],
    #                 "dtype": [np.int64],
    #                 "gen_fn": "Genfunc.randint",
    #             }
    #         ],
    #     ),
    # ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 2, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1,), (1,), (1,), (1,), (1,)], [(576,), (64,), (73728,), (128,), (294912,), (256,), (256,), (256,), (589824,), (256,), (1179648,), (512,), (512,), (512,), (2359296,), (512,), (1048576,), (512,), (512,), (512,), (524288,), (262144,), (1024,), (1024,), (524288,), (262144,), (1024,), (1024,), (131072,), (256,), (262144,), (262144,), (1024,), (1024,), (262144,), (262144,), (1024,), (1024,), (18944,), (37,), (256,), (256,), (512,), (512,), (512,), (512,)], [(26, 64, 256), (26, 64, 256)], [(64, 256), (64, 256), (64, 256), (64, 256)]],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                    "gen_policy": "gen_tensor_list_diff_shape",
                },
            ],
        ),
    ),

    'clamp_min': dict(
        name=["clamp_min"],
        interface=["torch"],
        para=dict(
            min=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(64,)],
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 1, 32, 100), (64, 512, 4, 26), (64, 128, 8, 25), (64, 64, 16, 50), (64, 256, 4, 26), (64, 512, 2, 27), (64, 256, 8, 25)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 1, 3, 3), (512, 512, 3, 3), (256, 128, 3, 3), (128, 64, 3, 3), (512, 256, 3, 3), (512, 512, 2, 2), (256, 256, 3, 3)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (512,), (256,), (128,), (512,), (512,), (256,)],
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
                    "shape": [(64,), (64,), (1, 32, 100)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64,), (64,), (1, 1, 1)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
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
                    "shape": [(128,), (512,), (256, 256, 3, 3), (512, 256, 3, 3), (256, 512), (1024, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(128,), (512,), (256, 256, 3, 3), (512, 256, 3, 3), (256, 512), (1024, 256)],
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
            other=[1, 1, 1, 64],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (64,)],
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
            value=[0, 0, 0, 0, 0, 0, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (128,), (1024, 256), (37, 512), (64, 1, 3, 3), (512, 512, 2, 2), (), (2, 64, 256)],
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
                    "shape": [(64, 256), (64, 256), (64, 256), (1664, 512), (1664, 512), (64, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1024, 256), (1024, 256), (1024, 256), (37, 512), (256, 512), (1024, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1024,), (1024,), (1024,), (37,), (256,), (1024,)],
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
            dim=[2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 26, 37)],
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
            stride=[(2, 1), (2, 1), (2, 2), (2, 2)],
            padding=[(0, 1), (0, 1), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1)],
            ceil_mode=[False, False, False, False],
            return_indices=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 256, 8, 25), (64, 512, 4, 26), (64, 128, 16, 50), (64, 64, 32, 100)],
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
                    "shape": [(256,), (37,), (256, 512), (1024, 512), (256, 128, 3, 3), (512, 512, 3, 3), ()],
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
                    "shape": [(64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256)],
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
                    "shape": [(64,), (128,), (128, 64, 3, 3), (256, 128, 3, 3), (37, 512), (1024, 512)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(64,), (128,), (128, 64, 3, 3), (256, 128, 3, 3), (37, 512), (1024, 512)],
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
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (37,), (512, 512, 3, 3), (128, 64, 3, 3), (37, 512), (1024, 512)],
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
            other=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,), (512,), (1024, 256), (256, 512), (512, 512, 2, 2), (256, 256, 3, 3)],
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
            mean=[0, 0, 0, 0, 0, 0, 0],
            std=[0.0340207, 0.0170103, 0.0208333, 0.0240563, 0.0584705, 0.0220971, 0.0147314],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 64, 3, 3), (512, 256, 3, 3), (256, 256, 3, 3), (256, 128, 3, 3), (64, 1, 3, 3), (512, 512, 2, 2), (512, 512, 3, 3)],
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
                    "shape": [(64, 512, 1, 26), (64, 256, 8, 25), (64, 64, 32, 100), (64, 512, 4, 26), (64, 128, 16, 50)],
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
                    "shape": [(64, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'sqrt': dict(
        name=["sqrt"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [False],
                    "shape": [(37,), (64,), (37, 512), (256, 512), (512, 512, 3, 3), (64, 1, 3, 3)],
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
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(64, 256), (64, 256)], [(64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512), (64, 512)], [(64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256), (64, 256)], [(1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100), (1, 32, 100)]],
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
                    "shape": [(1, 32, 100)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 1, 1)],
                    "dtype": [np.int64],
                    "gen_fn": "Genfunc.randint",
                },
            ],
        ),
    ),

    'tanh': dict(
        name=["tanh"],
        interface=["torch"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 256)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

    'tanh_case_2': dict(
        name=["tanh"],
        interface=["torch"],
        is_inplace=[True],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 256)],
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
            start=[0, 0],
            end=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(512,), (256,)],
                    "dtype": [np.float32],
                    "gen_fn": "Genfunc.randn",
                },
            ],
        ),
    ),

}
