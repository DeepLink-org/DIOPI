from ...config import Genfunc
from ...dtype import Dtype

detr_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 3, 962, 1009), (2, 64, 241, 253), (2, 64, 241, 253), (2, 64, 241, 253), (2, 256, 241, 253), (2, 256, 241, 253), (2, 128, 241, 253), (2, 128, 121, 127), (2, 256, 241, 253), (2, 512, 121, 127), (2, 128, 121, 127), (2, 512, 121, 127), (2, 256, 121, 127), (2, 256, 61, 64), (2, 512, 121, 127), (2, 1024, 61, 64), (2, 256, 61, 64), (2, 1024, 61, 64), (2, 512, 61, 64), (2, 512, 31, 32), (2, 1024, 61, 64), (2, 2048, 31, 32), (2, 512, 31, 32), (1, 3, 800, 1333), (1, 64, 200, 334), (1, 64, 200, 334), (1, 64, 200, 334), (1, 256, 200, 334), (1, 256, 200, 334), (1, 128, 200, 334), (1, 128, 100, 167), (1, 256, 200, 334), (1, 512, 100, 167), (1, 128, 100, 167), (1, 512, 100, 167), (1, 256, 100, 167), (1, 256, 50, 84), (1, 512, 100, 167), (1, 1024, 50, 84), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 512, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        para=dict(
            training=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 64, 481, 505), (2, 64, 241, 253), (2, 256, 241, 253), (2, 128, 241, 253), (2, 128, 121, 127), (2, 512, 121, 127), (2, 256, 121, 127), (2, 256, 61, 64), (2, 1024, 61, 64), (2, 512, 61, 64), (2, 512, 31, 32), (2, 2048, 31, 32), (1, 64, 400, 667), (1, 64, 200, 334), (1, 256, 200, 334), (1, 128, 200, 334), (1, 128, 100, 167), (1, 512, 100, 167), (1, 256, 100, 167), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 64, 481, 505), (2, 64, 241, 253), (2, 256, 241, 253), (2, 128, 241, 253), (2, 128, 121, 127), (2, 512, 121, 127), (2, 256, 121, 127), (2, 256, 61, 64), (2, 1024, 61, 64), (2, 512, 61, 64), (2, 512, 31, 32), (2, 2048, 31, 32), (992, 2, 2048), (100, 2, 2048), (6, 2, 100, 256), (1, 64, 400, 667), (1, 64, 200, 334), (1, 256, 200, 334), (1, 128, 200, 334), (1, 128, 100, 167), (1, 512, 100, 167), (1, 256, 100, 167), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42), (1050, 1, 2048), (100, 1, 2048), (6, 1, 100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=[False, False],
            return_indices=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 64, 481, 505), (1, 64, 400, 667)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 241, 253), (2, 512, 121, 127), (2, 1024, 61, 64), (2, 2048, 31, 32), (16, 992, 992), (16, 100, 992), (1, 256, 200, 334), (1, 512, 100, 167), (1, 1024, 50, 84), (1, 2048, 25, 42), (8, 1050, 1050), (8, 100, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 241, 253), (2, 512, 121, 127), (2, 1024, 61, 64), (2, 2048, 31, 32), (16, 1, 992), (16, 1, 992), (1, 256, 200, 334), (1, 512, 100, 167), (1, 1024, 50, 84), (1, 2048, 25, 42), (8, 1, 1050), (8, 1, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d_1': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1)],
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 2048, 31, 32), (1, 2048, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 2048, 1, 1), (256, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            size=[(31, 32), (25, 42)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 1, 962, 1009), (1, 1, 800, 1333)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32), (1, 25, 42)],
                    "dtype": [Dtype.int32],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'cumsum': dict(
        name=["cumsum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[1, 2, 1, 2],
            dtype=[Dtype.float32, Dtype.float32, Dtype.float32, Dtype.float32],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32), (2, 31, 32), (1, 25, 42), (1, 25, 42)],
                    "dtype": [Dtype.int32],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add_1': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.0, 1e-06, 1e-06, 1e-06, 0.0, 1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32), (2, 1, 32), (2, 31, 1), (), (1, 25, 42), (1, 1, 42), (1, 25, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32), (2, 31, 32), (2, 31, 32, 1), (14, 4), (100, 14), (1, 4), (100, 1), (200,), (1, 25, 42), (1, 25, 42), (1, 25, 42, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 1, 32), (2, 31, 1), (128,), (1, 4), (100, 14), (1, 4), (100, 1), (200,), (1, 1, 42), (1, 25, 1), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[6.283185307179586, 1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 0.1, 6.283185307179586, 1333, 800],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32), (100, 14), (100, 14), (100, 14), (100, 1), (100, 1), (100, 1), (), (1, 25, 42), (100, 2), (100, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[128],
            dtype=[Dtype.float32],
        ),
    ),

    'div_1': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_2': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[2],
            rounding_mode=['trunc'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[2, 0.5, 1.0, 0.5, 2.0, 5.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (100, 1), (), (200, 1), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_3': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[128, 5.656854249492381, 8, 5.656854249492381, 8, 8, 2, 2, 33.50000011920929, 15.00000011920929, 5.656854249492381, 8, 5.656854249492381, 8, 8],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (16, 992, 32), (2, 992, 992), (16, 100, 32), (2, 100, 100), (2, 100, 992), (14, 1), (1, 1), (), (), (8, 1050, 32), (1, 1050, 1050), (8, 100, 32), (1, 100, 100), (1, 100, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pow': dict(
        name=["pow"],
        interface=["torch.Tensor"],
        para=dict(
            exponent=[10000],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pow_1': dict(
        name=["pow"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["exponent"],
                    "shape": [(128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sin': dict(
        name=["sin"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32, 64), (1, 25, 42, 64)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cos': dict(
        name=["cos"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32, 64), (1, 25, 42, 64)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[4, 4],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((2, 31, 32, 64), (2, 31, 32, 64)), ((1, 25, 42, 64), (1, 25, 42, 64))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[3, -1, -1, -1, -1, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((2, 31, 32, 128), (2, 31, 32, 128)), ((14, 1), (14, 1), (14, 1), (14, 1)), ((100, 1), (100, 1), (100, 1), (100, 1)), ((1, 1), (1, 1), (1, 1), (1, 1)), ((200, 1), (200, 1), (200, 1), (200, 1)), ((1, 25, 42, 128), (1, 25, 42, 128))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'permute': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 3, 1, 2), (0, 3, 1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 31, 32, 256), (1, 25, 42, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute_1': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(2, 0, 1), (1, 2, 0), (2, 0, 1), (1, 2, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 992), (992, 2, 256), (1, 256, 1050), (1050, 1, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_2': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(992, 2, 256), (100, 2, 256), (14, 1), (100, 1), (100, 1), (100, 14), (1, 1), (100, 1), (200, 1), (200,), (), (1050, 1, 256), (100, 1, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(992, 2, 256), (100, 2, 256), (14, 1), (100, 1), (1, 14), (100, 14), (1, 1), (1, 1), (200, 1), (200,), (), (1050, 1, 256), (100, 1, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(992, 2, 256), (992, 2, 256), (992, 2, 2048), (100, 2, 256), (100, 2, 256), (100, 2, 2048), (6, 2, 100, 256), (6, 2, 100, 256), (6, 2, 100, 256), (1050, 1, 256), (1050, 1, 256), (1050, 1, 2048), (100, 1, 256), (100, 1, 256), (100, 1, 2048), (6, 1, 100, 256), (6, 1, 100, 256), (6, 1, 100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 256), (2048, 256), (256, 2048), (256, 256), (2048, 256), (256, 2048), (81, 256), (256, 256), (4, 256), (256, 256), (2048, 256), (256, 2048), (256, 256), (2048, 256), (256, 2048), (81, 256), (256, 256), (4, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (2048,), (256,), (256,), (2048,), (256,), (81,), (256,), (4,), (256,), (2048,), (256,), (256,), (2048,), (256,), (81,), (256,), (4,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'transpose': dict(
        name=["transpose"],
        interface=["torch.Tensor"],
        para=dict(
            dim0=[0, -2, 0, 0, -2, 0, 1, 0, -2, 0, 0, -2, 0, 1],
            dim1=[1, -1, 1, 1, -1, 1, 2, 1, -1, 1, 1, -1, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(992, 16, 32), (16, 992, 32), (16, 992, 32), (100, 16, 32), (16, 100, 32), (16, 100, 32), (6, 100, 2, 256), (1050, 8, 32), (8, 1050, 32), (8, 1050, 32), (100, 8, 32), (8, 100, 32), (8, 100, 32), (6, 100, 1, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(-1, 8, -1, -1), (-1, 8, -1, -1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 1, 1, 992), (1, 1, 1, 1050)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            value=[float("-inf"), float("-inf")],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 1, 992), (8, 1, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mask"],
                    "shape": [(16, 1, 992), (8, 1, 1050)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
                    "shape": [(16, 992, 32), (16, 992, 992), (16, 100, 32), (16, 100, 100), (16, 100, 32), (16, 100, 992), (8, 1050, 32), (8, 1050, 1050), (8, 100, 32), (8, 100, 100), (8, 100, 32), (8, 100, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mat2"],
                    "shape": [(16, 32, 992), (16, 992, 32), (16, 32, 100), (16, 100, 32), (16, 32, 992), (16, 992, 32), (8, 32, 1050), (8, 1050, 32), (8, 32, 100), (8, 100, 32), (8, 32, 1050), (8, 1050, 32)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, -1, -1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 992, 992), (16, 100, 100), (16, 100, 992), (8, 1050, 1050), (8, 100, 100), (8, 100, 1050), (100, 81)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 992, 992), (16, 100, 100), (16, 100, 992)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 8, 992, 992), (2, 8, 100, 100), (2, 8, 100, 992), (1, 8, 1050, 1050), (1, 8, 100, 100), (1, 8, 100, 1050)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout_1': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0],
            training=[True, True, True, True, True, True, True, False, False, False, False, False, False, False],
            inplace=[False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(992, 2, 256), (992, 2, 256), (992, 2, 2048), (100, 2, 256), (100, 2, 256), (100, 2, 2048), (6, 2, 100, 256), (1050, 1, 256), (1050, 1, 256), (1050, 1, 2048), (100, 1, 256), (100, 1, 256), (100, 1, 2048), (6, 1, 100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        para=dict(
            normalized_shape=[(256,), (256,), (256,), (256,)],
            eps=[1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(992, 2, 256), (100, 2, 256), (1050, 1, 256), (100, 1, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack_1': dict(
        name=["stack"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((100, 2, 256), (100, 2, 256), (100, 2, 256), (100, 2, 256), (100, 2, 256), (100, 2, 256)), ((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()), ((100, 1, 256), (100, 1, 256), (100, 1, 256), (100, 1, 256), (100, 1, 256), (100, 1, 256))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=["torch.Tensor"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(6, 2, 100, 4), (6, 1, 100, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'neg': dict(
        name=["neg"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 14), (100, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'split': dict(
        name=["split"],
        interface=["torch"],
        para=dict(
            split_size_or_sections=[(1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)],
            dim=[-1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensor"],
                    "shape": [(14, 4), (100, 4), (1, 4), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_1': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14, 1), (100, 1), (100,), (14,), (100, 14, 2), (100, 14), (1, 1), (1,), (100, 1, 2), (200, 1), (200,), (200, 2), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(14, 1), (100, 1), (100,), (14,), (100, 14, 2), (100, 14), (1, 1), (1,), (100, 1, 2), (200, 1), (200,), (200, 2), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cdist': dict(
        name=["cdist"],
        interface=["torch"],
        saved_args=dict(output=0),
        para=dict(
            p=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["x1"],
                    "requires_grad": [True],
                    "shape": [(100, 4), (100, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["x2"],
                    "shape": [(14, 4), (1, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 4), (100,), (14,), (100, 14), (1,), (100, 1), (200,), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (100,), (14,), (100, 14), (1,), (100, 1), (200,), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(100, 1, 2), (100, 14), (100, 1, 2), (100, 1), (200, 2), (200,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 14, 2), (1,), (1, 1, 2), (1,), (200, 2), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(100, 1, 2), (100, 1, 2), (200, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 14, 2), (1, 1, 2), (200, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[0, 0, 0],
            max=[None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 14, 2), (100, 1, 2), (200, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_3': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14,), (1,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(100,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'unique': dict(
        name=["unique"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14,), (86,), (1,), (99,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sub_2': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14,), (1,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'cat_1': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((100,), (100,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((100,), (100,)), ((100, 4), (100, 4)), ((100, 4), (100, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            ignore_index=[-100],
            reduction=['none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(200, 81)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(200,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sum_1': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200,), (200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_1': dict(
        name=["clamp"],
        interface=["torch"],
        para=dict(
            min=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gt_1': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(200, 4)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_3': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(200, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean_1': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_4': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (768, 256), (768,), (256, 256), (256,), (2048, 256), (2048,), (256, 2048), (256, 2048, 1, 1), (81, 256), (81,), (4, 256), (4,), (100, 256), (235,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_4': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_2': dict(
        name=["clamp"],
        interface=["torch"],
        para=dict(
            max=[1.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_3': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (768, 256), (768,), (256, 256), (256,), (2048, 256), (2048,), (256, 2048), (256, 2048, 1, 1), (81, 256), (81,), (4, 256), (4,), (100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adamw': dict(
        name=["adamw"],
        interface=["CustomizedTest"],
        para=dict(
            step=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            amsgrad=[False for i in range(15)],
            beta1=[0.9 for i in range(15)],
            beta2=[0.999 for i in range(15)],
            lr=[1e-05 for i in range(15)],
            weight_decay=[0.0001 for i in range(15)],
            eps=[1e-08 for i in range(15)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
                    "shape": [(128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adamw_1': dict(
        name=["adamw"],
        interface=["CustomizedTest"],
        para=dict(
            step=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            amsgrad=[False for i in range(13)],
            beta1=[0.9 for i in range(13)],
            beta2=[0.999 for i in range(13)],
            lr=[0.0001 for i in range(13)],
            weight_decay=[0.0001 for i in range(13)],
            eps=[1e-08 for i in range(13)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(768, 256), (768,), (256, 256), (256,), (2048, 256), (2048,), (256, 2048), (256, 2048, 1, 1), (81, 256), (81,), (4, 256), (4,), (100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
                    "shape": [(768, 256), (768,), (256, 256), (256,), (2048, 256), (2048,), (256, 2048), (256, 2048, 1, 1), (81, 256), (81,), (4, 256), (4,), (100, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch.Tensor"],
        para=dict(
            k=[100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_3': dict(
        name=["clamp"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            min=[0, 0],
            max=[1333, 800],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 2), (100, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_5': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(4,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
