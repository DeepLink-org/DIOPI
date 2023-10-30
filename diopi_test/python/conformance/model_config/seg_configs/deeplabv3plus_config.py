from ...config import Genfunc
from ...diopi_runtime import Dtype

deeplabv3plus_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2, 2048, 64, 128)],
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
            alpha=[0.0005, 0.0005, 0.0005, 0.0005, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048, 512, 1, 1), (512, 128, 1, 1), (1024,), (32,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2048, 512, 1, 1), (512, 128, 1, 1), (1024,), (32,), ()],
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
            alpha=[-0.00990037, -0.00994585, -0.00985197, -0.00998062],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (48,), (256, 512, 1, 1), (512, 512, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1024,), (48,), (256, 512, 1, 1), (512, 512, 3, 3)],
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

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[1.19209e-07, 0],
            alpha=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), ()],
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
                    "shape": [(2, 512, 128, 256), (2, 256, 128, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(512,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(512,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(512,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(512,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=["bernoulli"],
        interface=["torch.nn.functional"],
        para=dict(
            p=[0.9, 0.9],
            generator=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(2, 512, 1, 1), (2, 256, 1, 1)],
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
            dim=[1, 0, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(2, 512, 64, 128), (2, 512, 64, 128), (2, 512, 64, 128), (2, 512, 64, 128), (2, 512, 64, 128)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(864,), (32,), (32,), (9216,), (32,), (32,), (18432,), (64,), (64,), (4096,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (16384,), (64,), (64,), (36864,), (64,), (64,), (16384,), (256,), (256,), (32768,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (65536,), (128,), (128,), (147456,), (128,), (128,), (65536,), (512,), (512,), (131072,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (262144,), (256,), (256,), (589824,), (256,), (256,), (262144,), (1024,), (1024,), (524288,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (2097152,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (1048576,), (512,), (512,), (2359296,), (512,), (512,), (1048576,), (2048,), (2048,), (9728,), (19,), (1048576,), (512,), (512,), (1048576,), (512,), (512,), (18432,), (2048,), (2048,), (1048576,), (512,), (512,), (18432,), (2048,), (2048,), (1048576,), (512,), (512,), (18432,), (2048,), (2048,), (1048576,), (512,), (512,), (11796480,), (512,), (512,), (12288,), (48,), (48,), (5040,), (560,), (560,), (286720,), (512,), (512,), (4608,), (512,), (512,), (262144,), (512,), (512,), (4864,), (19,), (2359296,), (256,), (256,), (3,), (3,), (32,), (32,), (32,), (32,), (64,), (64,), (64,), (64,), (64,), (64,), (256,), (256,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (64,), (64,), (64,), (64,), (256,), (256,), (128,), (128,), (128,), (128,), (512,), (512,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (128,), (128,), (128,), (128,), (512,), (512,), (256,), (256,), (256,), (256,), (1024,), (1024,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (256,), (256,), (256,), (256,), (1024,), (1024,), (512,), (512,), (512,), (512,), (2048,), (2048,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (2048,), (2048,), (512,), (512,), (2048,), (2048,), (512,), (512,), (2048,), (2048,), (512,), (512,), (512,), (512,), (48,), (48,), (560,), (560,), (512,), (512,), (512,), (512,), (512,), (512,), (256,), (256,)], [(2, 512, 128, 256), (2, 48, 128, 256)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        interface=["torch.nn.functional"],
        para=dict(
            stride=[(2, 2), (1, 1)],
            padding=[(0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 512],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2, 256, 128, 256), (2, 512, 128, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(512, 256, 1, 1), (512, 1, 3, 3)],
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
                    "shape": [(3, 512, 1024)],
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
            other=[0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 512, 1, 1), (2, 256, 1, 1)],
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
            other=[1, 1, 1, 1048576],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (2, 512, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(1, 2, 512, 1024)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 2, 512, 1024)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 1, 0, 0, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(512,), (512,), (19,), (48,), (48,), ()],
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
                    "shape": [(2, 512, 1024), (2, 512, 1024), (1, 2, 512, 1024), (1, 2, 512, 1024)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["indices"],
                    "requires_grad":[False],
                    "shape": [(2, 512, 1024), (2, 512, 1024), (2, 512, 1024), (2, 512, 1024)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
                    "requires_grad":[True],
                    "shape": [(2, 19, 512, 1024)],
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
            kernel_size=[(3, 3)],
            stride=[(2, 2)],
            padding=[(1, 1)],
            dilation=[(1, 1)],
            ceil_mode=[False],
            return_indices=[True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2, 64, 256, 512)],
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
                    "shape": [(128, 256, 1, 1), (512, 2560, 3, 3), (2, 512, 1024), (32,), (48,), ()],
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
                    "shape": [(2, 512, 128, 256), (2, 512, 128, 256), (2, 256, 64, 128), (2, 256, 64, 128)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 512, 1, 1), (2, 512, 1, 1), (2, 256, 1, 1), (2, 256, 1, 1)],
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
            other=[0.9, 0.9, 9.79439e-05, 0.000101129],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 1024, 1, 1), (1024, 256, 1, 1), (1,), (1,)],
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
                    "shape": [(), (), (1024, 256, 1, 1), (256, 64, 1, 1), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(2, 512, 1024)],
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
                    "requires_grad":[False],
                    "shape": [(2, 512, 1024)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            mean=[0, 0, 0, 0],
            std=[0.0208333, 0.204124, 0.01, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(512, 2560, 3, 3), (48, 256, 1, 1), (19, 256, 1, 1), (19, 512, 1, 1)],
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
                    "shape": [(2, 1024, 64, 128), (2, 512, 128, 256)],
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
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1, 512, 1024), (1, 512, 1024)], [(1, 512, 1024), (1, 512, 1024)]],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(3, 512, 1024)],
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
                    "shape": [(1029552,), (1021283,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(2, 19, 512, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        interface=["torch.nn.functional"],
        para=dict(
            mode=['linear', 'linear', 'linear', 'linear'],
            size=[(128, 256), (512, 1024), (64, 128), (512, 1024)],
            align_corners=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2, 512, 64, 128), (2, 19, 128, 256), (2, 512, 1, 1), (2, 19, 64, 128)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'nll_loss2d': dict(
        name=["nll_loss2d"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(2, 19, 512, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["grad_output"],
                    "requires_grad":[False],
                    "shape": [(2, 512, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
