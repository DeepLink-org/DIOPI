from ...config import Genfunc
from ...diopi_runtime import Dtype

swin_transformer_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 1536, 7, 7), (64, 1536, 7, 7), (15, 1536, 7, 7), (16, 1536, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(64, 784, 384), (16, 3136, 192), (15, 4, 24, 49, 49), (64, 16, 12, 49, 49), (16, 48, 49, 49), (64, 24, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 784, 384), (16, 3136, 192), (1, 4, 1, 49, 49), (1, 16, 1, 49, 49), (1, 48, 49, 49), (1, 24, 49, 49)],
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
            alpha=[0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1536, 6144), (768, 1536), (1536,), (3072,), (3, 240, 12, 49, 32), (3, 256, 24, 49, 32), (15, 3136, 192), (15, 784, 384), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1536, 6144), (768, 1536), (1536,), (3072,), (3, 240, 12, 49, 32), (3, 256, 24, 49, 32), (15, 3136, 192), (15, 784, 384), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0, 0, 0, 0],
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(169, 6), (169, 48), (169, 24), (169, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[0.969565, 0.978261, 1e-08, 1e-08, 1e-08, 1e-08, 1e-06, 0, 1e-08],
            alpha=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 1, 1), (64, 1, 1), (169, 48), (1536, 1536), (768,), (1152,), (), (), (192, 3, 4, 4)],
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
            value=[-3.03453e-05, -4.075e-05, -6.20236e-06, -4.88168e-05, -2.48236e-05, -2.69619e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(384, 384), (1536, 384), (1152,), (4608,), (192, 3, 4, 4), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad":[False],
                    "shape": [(384, 384), (1536, 384), (1152,), (4608,), (192, 3, 4, 4), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad":[False],
                    "shape": [(384, 384), (1536, 384), (1152,), (4608,), (192, 3, 4, 4), (192, 3, 4, 4)],
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
            value=[0.001, 0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(1536, 1536), (1000, 1536), (192,), (1000,), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad":[False],
                    "shape": [(1536, 1536), (1000, 1536), (192,), (1000,), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad":[False],
                    "shape": [(1536, 1536), (1000, 1536), (192,), (1000,), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'argmax': dict(
        name=["argmax"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
            keepdim=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(64, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[True],
                    "shape": [(12288, 49, 49), (2880, 32, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mat2"],
                    "requires_grad":[False],
                    "shape": [(12288, 49, 32), (2880, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(9216,), (192,), (192,), (192,), (192,), (192,), (1014,), (110592,), (576,), (36864,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (192,), (192,), (1014,), (110592,), (576,), (36864,), (192,), (192,), (192,), (147456,), (768,), (147456,), (192,), (768,), (768,), (294912,), (384,), (384,), (2028,), (442368,), (1152,), (147456,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (384,), (384,), (2028,), (442368,), (1152,), (147456,), (384,), (384,), (384,), (589824,), (1536,), (589824,), (384,), (1536,), (1536,), (1179648,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,), (3072,), (2359296,), (768,), (768,), (768,), (4056,), (1769472,), (2304,), (589824,), (768,), (768,), (768,), (2359296,)], [(768,), (3072,), (3072,), (4718592,), (1536,), (1536,), (8112,), (7077888,), (4608,), (2359296,), (1536,), (1536,), (1536,), (9437184,), (6144,), (9437184,), (1536,), (1536,), (1536,), (8112,), (7077888,), (4608,), (2359296,), (1536,), (1536,), (1536,), (9437184,), (6144,), (9437184,), (1536,), (1536,), (1536,), (1536000,), (1000,), (3,), (3,)], [(2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,), (2401,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        atol=1e-04,
        rtol=1e-05,
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[-2, -2, -2, -2],
            max=[2, 2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(169, 48), (169, 12), (169, 6), (169, 24)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            min=[None],
            max=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'col2_im': dict(
        name=["col2_im"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(28, 28), (14, 14), (56, 56), (14, 14), (56, 56), (28, 28)],
            kernel_size=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            stride=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(64, 1536, 196), (64, 3072, 49), (64, 768, 784), (15, 3072, 49), (15, 768, 784), (15, 1536, 196)],
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
            stride=[(4, 4), (4, 4), (4, 4)],
            padding=[(0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(64, 3, 224, 224), (16, 3, 224, 224), (15, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(192, 3, 4, 4), (192, 3, 4, 4), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(192,), (192,), (192,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(64, 3, 224, 224), (15, 3, 224, 224), (16, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1)],
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
            other=[0.864923, 0.631709, 0.870895, 0.962389, 0.999873, 0.977159],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1536, 6144), (1536, 384), (1152,), (2304,), (192, 3, 4, 4), (192, 3, 4, 4)],
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
            other=[15, 1, 0.991304, 0.904348],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (64, 784, 384), (15, 49, 1536)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 49, 49), (4, 49, 49), (64, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'erf': dict(
        name=["erf"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(169, 12), (169, 6), (169, 48), (169, 24)],
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
            value=[0, 0, 8, 1, 0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4608, 1536), (1536, 3072), (), (), (1, 56, 56, 1), (1, 14, 14, 1), (4608,), (1536,), (3, 960, 6, 49, 32), (3, 15, 48, 49, 32)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'flip': dict(
        name=["flip"],
        interface=["torch"],
        para=dict(
            dims=[(1,), (1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 224, 224), (15, 3, 224, 224), (16, 3, 224, 224)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
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
                    "requires_grad":[False],
                    "shape": [(64, 1, 1), (15, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gelu': dict(
        name=["gelu"],
        interface=["torch.nn.functional"],
        para=dict(
            approximate=['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 3136, 768), (64, 3136, 768), (64, 196, 3072), (64, 196, 3072), (64, 784, 1536), (64, 784, 1536), (64, 49, 6144), (64, 49, 6144), (15, 784, 1536), (16, 49, 6144), (15, 3136, 768), (16, 784, 1536), (15, 49, 6144), (16, 196, 3072), (16, 3136, 768), (15, 196, 3072)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'im2_col': dict(
        name=["im2_col"],
        interface=["torch.nn.functional"],
        para=dict(
            kernel_size=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            stride=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(64, 768, 14, 14), (64, 768, 14, 14), (64, 192, 56, 56), (64, 192, 56, 56), (64, 384, 28, 28), (64, 384, 28, 28), (16, 192, 56, 56), (16, 768, 14, 14), (15, 768, 14, 14), (16, 384, 28, 28), (15, 192, 56, 56), (15, 384, 28, 28)],
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
                    "requires_grad":[True],
                    "shape": [(169, 48), (169, 6), (169, 24), (169, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["indices"],
                    "requires_grad":[False],
                    "shape": [(2401,), (2401,), (2401,), (2401,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(Genfunc.randint, low=-169, high=169),
                },
            ],
        ),
    ),

    'index_put': dict(
        name=["index_put"],
        interface=["CustomizedTest"],
        para=dict(
            accumulate=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(169, 12), (169, 48), (169, 24), (169, 6)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["indices1"],
                    "requires_grad":[False],
                    "shape": [(2401,), (2401,), (2401,), (2401,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(Genfunc.randint, low=-169, high=169),
                },
                {
                    "ins": ["values"],
                    "requires_grad":[False],
                    "shape": [(2401, 12), (2401, 48), (2401, 24), (2401, 6)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        interface=["torch.nn.functional"],
        para=dict(
            normalized_shape=[(1536,), (3072,)],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(15, 196, 1536), (16, 49, 3072)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1536,), (3072,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(1536,), (3072,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["save_mean"],
                    "requires_grad":[False],
                    "shape": [(15, 196, 1), (16, 49, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["save_invstd"],
                    "requires_grad":[False],
                    "shape": [(15, 196, 1), (16, 49, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[True],
                    "shape": [(64, 49, 768), (15, 49, 1536), (64, 1536), (16, 1536)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(768, 768), (4608, 1536), (1000, 1536), (1000, 1536)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(768,), (4608,), (1000,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[True],
                    "shape": [(64, 1000), (15, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        interface=["torch"],
        para=dict(
            value=[-100, 0, -100, 0, 0, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(4, 49, 49), (4, 49, 49), (64, 49, 49), (64, 49, 49), (16, 49, 49), (16, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mask"],
                    "requires_grad":[False],
                    "shape": [(4, 49, 49), (4, 49, 49), (64, 49, 49), (64, 49, 49), (16, 49, 49), (16, 49, 49)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
                    "shape": [(3072,), (384,), (1536, 3072), (384, 1536), (192, 3, 4, 4), ()],
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
                    "shape": [(15, 49, 1536), (64, 3136, 192)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 1, 1), (64, 1, 1)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6144,), (1000,), (576, 192), (768, 3072), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.999998, 0.999997, 0.999, 1, 0.999999, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(384, 384), (3072, 768), (768,), (1000,), (192, 3, 4, 4), (192, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_4': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 0.176777, 0.176777, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (576,), (768,), (1024, 12, 49, 32), (64, 48, 49, 32), (768, 768), (1536, 3072)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[True],
                    "shape": [(64, 1000), (15, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(64,), (15,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 49, 49), (4, 49, 49), (64, 49, 49)],
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
            p=[2, 2, 2, 2, 2],
            dim=[(0,), (0,), (0, 1), (0, 1), (0, 1, 2, 3)],
            keepdim=[False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(1536,), (384,), (3072, 768), (1000, 1536), (192, 3, 4, 4)],
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
            mean=[0],
            std=[0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000, 1536)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[False],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'roll': dict(
        name=["roll"],
        interface=["torch"],
        para=dict(
            shifts=[(-3, -3), (-3, -3)],
            dims=[(1, 2), (1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(64, 14, 14, 768), (64, 28, 28, 384)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, -1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(256, 12, 49, 49), (1024, 6, 49, 49), (16, 1000), (64, 1000)],
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
                    "shape": [(1000, 1536), (169, 24), (1536,), (384,), (192, 3, 4, 4)],
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
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()]],
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
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 1, 49), (64, 3, 224, 224), (64, 1, 49), (16, 1, 49), (15, 3, 224, 224), (16, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(4, 49, 1), (3, 1, 1), (64, 49, 1), (16, 49, 1), (3, 1, 1), (3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[[0], [0], [0], [0], [0], [0], [0], None, [0], [0], [0], [0], [0], [0], [0], None],
            keepdim=[True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, False],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 24, 49, 49), (256, 24, 49, 49), (4096, 6, 49, 49), (4096, 6, 49, 49), (1024, 12, 49, 49), (1024, 12, 49, 49), (64, 48, 49, 49), (64,), (15, 48, 49, 49), (60, 24, 49, 49), (60, 24, 49, 49), (240, 12, 49, 49), (240, 12, 49, 49), (960, 6, 49, 49), (960, 6, 49, 49), (15,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            start=[-1, -1, -1, 0, -1, 0],
            end=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(169, 48), (169, 12), (169, 24), (64, 1, 1), (169, 6), (15, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
