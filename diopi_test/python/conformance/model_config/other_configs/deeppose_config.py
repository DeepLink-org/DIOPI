from ...config import Genfunc
from ...diopi_runtime import Dtype

deeppose_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[1, 0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 2048, 8, 6), (512, 512, 3, 3), (64,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 2048, 8, 6), (512, 512, 3, 3), (64,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_2': dict(
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
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[1e-08, 1e-08, 1e-08, 1e-08, 0],
            alpha=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 256, 1, 1), (2048, 512, 1, 1), (64,), (512,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[-0.000368869, -0.000116617, -0.000416917, -0.000142642],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2048, 1024, 1, 1), (512, 1024, 1, 1), (2048,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad":[False],
                    "shape": [(2048, 1024, 1, 1), (512, 1024, 1, 1), (2048,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad":[False],
                    "shape": [(2048, 1024, 1, 1), (512, 1024, 1, 1), (2048,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'addcmul': dict(
        name=["addcmul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            value=[0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(64, 64, 3, 3), (1024, 512, 1, 1), (17,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad":[False],
                    "shape": [(64, 64, 3, 3), (1024, 512, 1, 1), (17,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad":[False],
                    "shape": [(64, 64, 3, 3), (1024, 512, 1, 1), (17,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            momentum=[0.1, 0.1],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 256, 64, 48), (53, 128, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
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
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17)], [(1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17), (1, 17)], [(9408,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (8388608,), (256,), (256,), (1048576,), (256,), (256,), (1048576,), (256,), (256,), (4352,), (17,), (3,), (3,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (256,), (256,), (256,), (256,), (256,), (256,)]],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        interface=["torch.nn.functional"],
        para=dict(
            bias=[None, None, None, None, None, None],
            stride=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            output_padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            groups=[1, 1, 1, 1, 1, 1],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 256, 32, 24), (64, 2048, 8, 6), (64, 256, 16, 12), (53, 2048, 8, 6), (53, 256, 32, 24), (53, 256, 16, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(256, 256, 4, 4), (2048, 256, 4, 4), (256, 256, 4, 4), (2048, 256, 4, 4), (256, 256, 4, 4), (256, 256, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 512, 32, 24), (53, 64, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1024, 512, 1, 1), (256, 64, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[False],
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
                    "shape": [(3, 256, 192)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_case_2': dict(
        name=["div"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.866803, 0.943737, 0.377881, 0.546122],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(512, 256, 1, 1), (64, 3, 7, 7), (512,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_case_3': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 3342336, 2767872],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (64, 17, 64, 48), (53, 17, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 0, 0, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(17,), (256,), (128, 128, 3, 3), (1024, 512, 1, 1), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[False],
                    "shape": [(3, 256, 192)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["indices"],
                    "requires_grad":[False],
                    "shape": [(105,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=-3, high=3),
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            reduction=['none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(53, 17, 64, 48), (64, 17, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(53, 17, 64, 48), (64, 17, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[True],
                    "shape": [(64, 64, 128, 96), (53, 64, 128, 96)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(64, 256, 1, 1), (512, 2048, 1, 1), (), (), (1024,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(53, 17, 64, 48), (53, 17, 64, 48), (64, 17, 64, 48), (64, 17, 64, 48)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(53, 17, 1, 1), (53, 17, 1, 1), (64, 17, 1, 1), (64, 17, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.999, 0.9, 0.9, 0.999],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 512, 1, 1), (17, 256, 1, 1), (17,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 256, 4, 4), (2048, 512, 1, 1), (256,), (17,), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            std=[0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 256, 4, 4), (17, 256, 1, 1), (256, 256, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(53, 2048, 8, 6), (64, 64, 128, 96)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[False],
                    "shape": [(128, 128, 3, 3), (128, 256, 1, 1), (512,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
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
                    "shape": [[(3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192)], [(17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48)], [(17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48), (17, 64, 48)], [(3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192), (3, 256, 192)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
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
                    "shape": [(3, 256, 192)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
