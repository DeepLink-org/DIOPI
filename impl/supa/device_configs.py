# Copyright (c) 2023, DeepLink.

import numpy as np
from skip import Skip

__Skip=lambda *args: [ Skip(x) for x in args]

device_configs = {

    'batch_norm_no_contiguous': dict(
        name=["batch_norm"],
        dtype=__Skip(np.float32, np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (32, 16)),
                    # "stride":((2000000, 230400, 7200, 120, 2), (1, 2048, 2, 64), (1, 56, 2), (20, 1)),
                },
                # {
                #     "ins": ["running_mean"],
                #     "stride":((4, ), None, None, None),
                #     "shape": ((8, ), (64, ), None, (16, )),
                # },
                # {
                #     "ins": ["running_var"],
                #     "shape": ((8, ), (64, ), None, (16, )),
                # },
                # {
                #     "ins": ["weight", "bias"],
                #     "shape": ((8, ), (64, ), (96, ), (16, )),
                # },
            ]
        ),
    ),

    'cast_dtype': dict(
        name=["cast_dtype"],
        interface=['CustomizedTest'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip(()),
                    "dtype":__Skip(np.float16, np.int16, np.int8, np.uint8)
                },
                {
                    "ins": ['out'],
                    # "shape": [(32, 64,), (128, 24, 32), (16, 8,), (24, 12,), (),
                    #           (0,), (4, 0), (5, 0, 7)],
                    "dtype":__Skip(np.float16, np.int16)
                }
            ]
        ),
    ),

    'copy': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=__Skip(np.float32, np.float64, np.float16, np.bool_,
               np.int64, np.int32, np.int16, np.int8, np.uint8),
        # tensor_para=dict(
        #     gen_fn='Genfunc.randn',
        #     args=[
        #         {
        #             "ins": ["input"],
        #             "shape": ((), (8,), (12,), (192, 147), (1, 1, 384), (2, 1, 38, 45),
        #                       (0,), (0, 12,), (12, 0, 9)),
        #             "no_contiguous": [True],
        #         },
        #         {
        #             "ins": ["other"],
        #             "shape": ((), (), (12,), (147, 1), (384, 1, 1), (45, 38, 1, 2),
        #                       (0,), (12, 0), (9, 0, 12)),
        #         },
        #     ]
        # )
    ),

    'copy_different_dtype': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=__Skip(np.float32, np.float64, np.float16, np.bool_,
               np.int64, np.int32, np.int16, np.int8, np.uint8),
        # tensor_para=dict(
        #     gen_fn='Genfunc.randn',
        #     args=[
        #         {
        #             "ins": ["input"],
        #             "shape": ((192, 147), (1, 1, 384), (2, 1, 38, 45), (100, 100)),
        #             "dtype": [np.float32, np.float64, np.float16, np.bool_,
        #                       np.int64, np.int32, np.int16, np.int8, np.uint8],
        #             "no_contiguous": [True],
        #         },
        #         {
        #             "ins": ["other"],
        #             "dtype": [np.float64, np.int64, np.float16, np.float16,
        #                       np.int32, np.float32, np.uint8, np.uint8, np.uint8],
        #             "shape": ((147, 1), (384, 1, 1), (45, 38, 1, 2), (1, 100)),
        #         },
        #     ]
        # )
    ),
    'copy_broadcast': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=__Skip(np.float32, np.float64),
        # tensor_para=dict(
        #     args=[
        #         {
        #             "ins": ["input"],
        #             "shape": ((8,), (12, 2), (192, 147, 2), (6, 5, 384), (2, 12, 38, 45, 3),
        #                        (0, 2), (0, 12,), (12, 0, 9, 2)),
        #             "no_contiguous": [True],
        #         },
        #         {
        #             "ins": ["other"],
        #             "shape": ((1,), (12,), (1, 147), (6, 1, 384), (2, 1, 38, 45),
        #                       (1,), (0, 1,), (12, 0, 1)),
        #             "no_contiguous": [True],
        #         },
        #     ]
        # )
    ),
    'cumsum': dict(
        name=["cumsum"],
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            dim=__Skip(2, 0),
        ),
        dtype = __Skip(np.float64, np.float16, np.int16, np.int32,
                            np.int64, np.uint8, np.int8, np.bool_),

        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (12,), (2, 22, 33), (2, 2, 10, 16), (2, 2, 20),
                              (0,), (5, 0), (4, 0, 12))
                },
            ],
        ),
    ),
    'flip': dict(
        name=['flip'],
        dtype=__Skip(np.float64, np.float16, np.int16, np.int32,
                     np.int64, np.uint8, np.int8, np.bool_),
        para=dict(
            dims=__Skip((0, 2, -1, -3,))
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": __Skip((), (12,13,14), (2,3,4,10,12), (0,), (12,0), (2,0,7))
                },
            ],
        ),
    ),
    'pointwise_op': dict(
        name=['abs', 'cos', 'erf', 'erfinv', 'exp', 'floor',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
        dtype=__Skip(np.float16, np.float64),
        tensor_para=dict(
            args=[
                {
                    "ins" : ["input"],
                    "shape" : __Skip((2, 31, 512, 6, 40))
                },
            ]
        )
    ),

    'pointwise_op_int_without_inplace': dict (
        name=['abs', 'cos', 'erf', 'erfinv', 'exp', 'floor',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
        dtype=__Skip(np.float16, np.int32, np.int16, np.int64, np.int8),
    ),

    'pointwise_op_without_inplace_zero': dict (
        name=['abs', 'sign', 'exp', 'sqrt', 'logical_not', 'rsqrt'],
        dtype= __Skip(np.float16, np.float64, np.int16,
               np.int32, np.int64, np.uint8, np.int8, np.bool_),
    ),

    'pointwise_op_zero': dict(
        name=['abs', 'exp', 'floor', 'neg', 'sqrt', 'logical_not', 'rsqrt', 'ceil'],
        dtype= __Skip(np.float16, np.float64, np.int16,
               np.int32, np.int64, np.uint8, np.int8, np.bool_),
    ),

    'pointwise_op_uint8': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp',
        #       'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        name=['abs', 'cos', 'erf', 'exp',
              'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        dtype=__Skip(np.uint8),
    ),

    'pointwise_op_bool': dict(
        # name=['abs', 'cos', 'erf', 'erfinv', 'exp',
        #       'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
        name=['abs', 'cos', 'erf', 'exp', 'sin', 'asin', 'sqrt', 'rsqrt', 'atan', 'logical_not'],
        dtype=__Skip(np.uint8),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'pointwise_op_mask': dict(
        name=['logical_not', 'bitwise_not'],
        dtype=__Skip(np.int16, np.int64, np.uint8, np.int8, np.bool_),
    ),

    'pointwise_op_abs_input': dict(
        name=['log', 'log2', 'log10', 'sqrt', 'rsqrt'],
        dtype=__Skip(np.float16, np.float64),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((2, 31, 512, 6, 40))
                },
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        dtype=__Skip(np.float16, np.int64, np.int32, np.int16, np.int8, np.uint8),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip(()),
                },
            ],
        ),
    ),
    'pointwise_binary_broadcast': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (2, 32, 130, 130), (8,16,1)),
                },
                {
                    "ins": ['other'],
                    "shape": __Skip((), (16, 0), (16, 0,), (0, 16)),
                },
            ],
        ),
    ),

    'pointwise_binary_broadcast_inplace': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((5, 2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": __Skip(()),
                },
            ],
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['add', 'mul', 'div', 'eq',
              'ne', 'le', 'lt', 'gt', 'ge'],
        # para=dict(
        #     other=[0, -1, 0.028, 2.232, 1, True, False],
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (1024,), (384, 128), (2, 64, 128), (128, 64, 3, 3), (128, 32, 2, 2), (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'pointwise_binary_scalar_div_zero': dict(
        name=['div'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((1024, ), (384, 128), (128, 64, 3, 3),(2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add'],
        para=dict(
            alpha=__Skip(0, -2, 2.0, 4, 1, 0.234, -2.123)
        ),
    ),


    'pointwise_binary_with_alpha': dict(
        name=['add', 'sub'],
        para=dict(
            alpha=__Skip(-2, 2.0),
        ),
    ),

    'pointwise_binary_with_alpha_bool': dict(
        name=['add'],
        para=dict(
            alpha=__Skip(True, False)
        ),
        tensor_para=dict(
            args=[
                # {
                #     "ins": ['input'],
                #     "shape": ((2, 3),
                #               (2, 2, 4, 3)),
                # },
                # {
                #     "ins": ['other'],
                #     "shape": ((1,), (1,)),
                # }
            ],
        ),
    ),
    'pointwise_binary_diff_dtype_without_bool': dict(
        # name=['sub', 'div'],
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((1024, ),),
                },
            ],
        ),
    ),

    'pointwise_binary_diff_dtype': dict(
        # name=['add', 'mul', 'eq', 'ne', 'le',
        #       'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        name=['mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype":__Skip(np.float64, np.float32, np.float16,
                             np.int64, np.int32, np.int16,
                             np.int8, np.uint8, np.bool_),
                },
            ],
        ),
    ),

    'pointwise_binary_diff_dtype_inplace': dict(
        # name=['add', 'mul', 'eq', 'ne', 'le',
        #       'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        name=['eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype":__Skip(np.float64, np.float32, np.float16,
                             np.int32, np.float64, np.float64,
                             np.int8, np.float32, np.int8),
                },
            ],
        ),
    ),

    'pointwise_binary_dtype_bool': dict(
        name=['add', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge',
              'logical_and', 'logical_or'],
        # dtype=[np.bool_],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
            ],
        ),
    ),



    'div': dict(
        name=['div'],
        dtype=__Skip(np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((),),
                },
            ],
        ),
    ),

    'div_broadcast': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'div_diff_dtype_inplace': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype":__Skip(np.float64, np.float32, np.float16),
                },
            ],
        ),
    ),

    'div_rounding_mode': dict(
        name=['div'],
        # para=dict(
        #     rounding_mode=['floor', None, 'floor', 'trunc', 'floor'],
        # ),
        dtype=__Skip(np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((),  (384, 128), (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'div_dtype_int_and_bool': dict(
        name=['div'],
        dtype=__Skip(np.int8, np.int16, np.int32, np.int64, np.uint8, np.bool_),
    ),


    'embedding': dict(
        name=["embedding"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": __Skip(()),
                    "dtype": __Skip(np.int64, np.int64),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    # "shape": ((10, 3), (10, 2), (93, 512), (20, 2), (16, 8),
                    #           (15, 3), (20, 3), (10, 5), (10, 4)),
                    "dtype": __Skip(np.float16, np.float64),
                },
            ],
        ),
    ),
    'embedding_forward': dict(
        name=["embedding"],
        dtype=__Skip(np.float16, np.int64, np.int32, np.int16, np.int8, np.uint8),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # "shape": ((1, 32),),
                },
                {
                    "ins": ["weight"],
                    # "shape": ((10, 0),),
                },
            ],
        ),
    ),
    'expand': dict(
        name=['expand'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((),(4, 1, 6, 8),(8,), (0,), (12, 0), (4, 0, 1)
                    # , (8,), (60800, 1), (100, 1), (70, 1, 2), (3, 1), (4, 1, 6, 8),
                            #   (0,), (12, 0), (4, 0, 1)
                                ),
                    "dtype": __Skip(np.bool_, np.float16, np.float64,
                              np.int64, np.int16, np.int8, np.uint8),
                },
            ],
        ),
    ),
    'fill': dict(
        name=["fill_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip(()),
                    "dtype": __Skip(np.float16),
                },
            ],
        ),
    ),

    'fill_not_float': dict(
        name=["fill_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip(()),
                },
            ],
        ),
    ),

    'gather': dict(
        name=['gather'],
        para=dict(
            dim=__Skip(0, -1),
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((8,), (9,), (2, 0), (5, 0, 9)),
                    "dtype": __Skip(np.float64, np.float16),
                },
                # {
                    # "ins": ['index'],
                    # "dtype": [np.int64],
                # },
            ],
        ),
    ),

    'gather_0dim': dict(
        name=['gather'],
        para=dict(
            dim=__Skip(0, -1),
        ),
    ),

    'gather_not_float': dict(
        name=['gather'],
        para=dict(
            dim=__Skip(0, -1),
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((8,), (9,), (2, 0), (5, 0, 9)),
                    "dtype": __Skip(np.int16, np.int64, np.uint8, np.int8, np.bool_),
                },
            ],
        ),
    ),

    # FIXME linear输入指定shape报错
    'linear': dict(
        name=["linear"],
        atol=1e-3,
        rtol=1e-4,
        atol_half=1e-1,
        rtol_half=1e-1,
        dtype=__Skip(np.float16, np.float64),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((8,), (8,), (2, 512), (128, 49, 128), (6, 2, 100, 256),
                    #           (2, 6, 16, 8), (2, 31, 6, 40, 512), (2, 16, 8, 32, 7), (0,), (0,), (16, 8)),
                    "shape": __Skip((2, 512), (2, 31, 6, 40, 512), (16, 8))
                },
            ]
        ),
    ),

    'index_select': dict(
        name=["index_select"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": __Skip((12, 0), (2, 0, 9)),
                    "dtype": __Skip(np.float16),
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    # "shape": ((10,), (3,), (5,), (2,), (30,),
                    #           (12,), (7,)),
                    # "dtype": [np.int64, np.int32, np.int64],
                },
            ]
        ),
    ),

    'index_select_not_float': dict(
        name=["index_select"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [False],
                    "shape": __Skip((12, 0), (2, 0, 15)),
                    "dtype": __Skip(np.int16, np.uint8, np.int8),
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    # "shape": ((20,), (10,), (5,), (100,), (10,),
                    #           (20,), (7,)),
                    # "dtype": [np.int32, np.int64, np.int32, np.int64, np.int32, np.int64],
                },
            ]
        ),
    ),

    # FIXME stack输入size为0的张量报错
    'join': dict(
        name=['cat', 'stack'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-4,
        # para=dict(
        #     # dim=[-1, 1, 0, 2, 1, 1, -1, 1, -2],
        #     dim=[-1, 1, 0, 2, 1, 1, -1, 1],
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "shape": __Skip((3, ), (512, 4), (0, 50, 76), (2, 512, 8, 8), (1, 64, 4, 56, 56), (0,), (16, 0)),
                    "dtype": __Skip(np.float16, np.int16, np.float64),
                },
            ],
            seq_name='tensors',
        ),
    ),
    'join_int': dict(
        name=['cat', 'stack'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    # "shape": ((3, ), (512, 4),
                    #           (0, 50, 76), (2, 31, 512),
                    #           (2, 512, 8, 8), (1, 64, 4, 56, 56),
                    #           (0,), (16, 0), (8, 0, 2)),
                    "shape": __Skip((3,), (512, 4), (2, 31, 512), (0, 50, 76), (2, 512, 8, 8),(1, 64, 4, 56, 56), (0,), (16, 0)),
                    # "dtype": [np.int64, np.uint8, np.int8, np.bool_, np.int32],

                },
            ],
        ),
    ),


    'cat_diff_size': dict(
        name=['cat'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "requires_grad": [True],
                    "shape": __Skip(((8,), (16,),),
                            #   ((2, 8,), (16, 8,), (3, 8,), (4, 8,), (1, 8,)),
                              ((3, 16, 8,), (3, 2, 8,), (3, 7, 8,)),
                              ((2, 512, 8, 8), (2, 128, 8, 8), (2, 2, 8, 8), (2, 1, 8, 8)),
                              ((2, 31, 0), (2, 31, 512), (2, 31, 128)),
                            ),
                    "dtype": __Skip(np.float16, np.float64, np.int16)
                },
            ],
            seq_name='tensors',
        ),
    ),

    'mm': dict(
        name=['mm'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((8, 48), (4, 128), (256, 8)),
                    "dtype": __Skip(np.float64, np.float32, np.float16),
                },
                # {
                #     "ins": ['mat2'],
                #     "shape": ((48, 128), (128, 128), (8, 1)),
                #     "dtype": [np.float32, np.float64, np.float16],
                # },
            ],
        ),
    ),

    'mm_diff_dtype': dict(
        name=['mm'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((8, 0), (0, 128), (256, 8)),
                    "dtype": __Skip(np.float32, np.float16, np.float64)
                },
                # {
                #     "ins": ['mat2'],
                #     "shape": ((0, 128), (128, 128), (8, 0)),
                #     "dtype": [np.float16, np.float64, np.float32],
                # },
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        dtype=__Skip(np.float16, np.float32, np.float64),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": ((16, 726, 32), (16, 100, 100), (9, 5, 5),
                    #           (0, 12, 16), (4, 0, 6), (4, 9, 0), (5, 8, 13)),
                },
                {
                    "ins": ['mat2'],
                    # "shape": ((16, 32, 726), (16, 100, 32), (9, 5, 10),
                    #           (0, 16, 7), (4, 6, 8), (4, 0, 12), (5, 13, 0)),
                },
            ],
        ),
    ),

    'pow': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (20267, 80), (2, 512, 38, 38), (0, 8), (7, 0, 9)),
                    "dtype": __Skip(np.float16, np.float64),
                }
            ],
        ),
    ),

    'pow_int': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": __Skip(np.int16, np.int32, np.int64,
                              np.int8, np.uint8),
                }
            ],
        ),
    ),

    'pow_bool': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (0, 8)),
                    "dtype": __Skip(np.bool_),
                }
            ],
        ),
    ),

    'pow_tensor': dict(
        name=['pow'],
        dtype=__Skip(np.float16, np.int32, np.float64,
               np.int16,np.int64,
               np.int8, np.uint8),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (0, 4), (9, 0, 3)),
                },
            ],
        ),
    ),

    'pow_tensor_only_0_1': dict(
        name=['pow'],
        dtype=__Skip(np.int16, np.int64,
               np.int8, np.uint8),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (1,), (20267, 80), (2, 128, 3072), (2, 512, 38, 38), (0, 4), (9, 0, 3)),
                },
            ],
        ),
    ),

    'pow_broadcast': dict(
        name=['pow'],
        dtype=__Skip(np.float64, np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,),(2, 1, 128),(2, 64, 1, 128), (2, 32, 130, 130), (8, 16, 1)),
                },
            ],
        ),
    ),

    'pow_broadcast_inplace': dict(
        name=['pow'],
        dtype=__Skip(np.float64, np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((16, 0,),  (2, 1024), (2, 384, 128) , (2, 64, 16, 128), (5, 2, 32, 130, 130), (8, 16, 0), (32, 0, 16)),
                },
            ],
        ),
    ),

    'pow_diff_dtype_cast': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype":__Skip(np.int64, np.int32, np.int16,
                             np.bool_, np.bool_, np.bool_, np.bool_),
                },
            ],
        ),
    ),

    # FIXME pow的input与exponent输入uint8和int8，结果不一致
    'pow_diff_dtype': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "dtype":[np.float64, np.float32, np.float16,
                    #          np.int32, np.float64, np.float64,
                    #          np.int8, np.float32, np.uint8],
                    "dtype":__Skip(np.float64, np.float16,
                             np.int32, np.float64,
                             np.int16, np.int64),
                },
                {
                    "ins": ['exponent'],
                    "dtype":__Skip(np.int32, np.uint8, np.bool_,
                             np.int64, np.float16, np.float64,
                             np.bool_, np.uint8, np.bool_),
                },
            ],
        ),
    ),

    'pow_input_scalar': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "shape": __Skip((), (0,), (0, 4), (9, 0, 6)),
                    "dtype":__Skip(np.float16, np.float64,
                              np.int16, np.int32, np.int64,
                              np.int8, np.uint8, np.bool_),
                }
            ],
        ),
    ),

    'pow_input_scalar_bool': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "dtype": __Skip(np.float16, np.float64,
                              np.int16, np.int32, np.int64,
                              np.int8, np.uint8),
                }
            ],
        ),
    ),

    # 'sgn': dict(
    #     name=['sgn'],
    #     interface=['torch'],
    #     dtype=__Skip(np.complex64, np.complex128)

    # ),

    # 'sgn_zero': dict(
    #     name=['sgn'],
    #     interface=['torch'],
    #     dtype=__Skip(np.complex64, np.complex128)
    # ),

    'silu': dict(
        name=["silu"],
        atol=1e-3,
        rtol=1e-4,
        dtype=__Skip(np.float64, np.int64),
    ),

    'soft_max': dict(
        name=["softmax"],
        # para=dict(
        #     dim=[-1, 0, -1, 1, 0, -1, -1, 1, -2],
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (16,), (0, 12), (16, 0, 7), (2, 128, 24), (8, 16, 49, 49)),
                    "dtype": __Skip(np.float16, np.float64),
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        dtype=__Skip(np.float16, np.float64,  np.float32, np.int16, np.int64, np.uint8, np.int8),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (11400,), (8, 12, 9),  (4, 4, 16, 20), (4, 4, 16, 2, 20),
                    (24180,), (0,), (12, 0), (4, 0, 5)
                    ),
                },
            ],
        ),
    ),

    'sort_same_value': dict(
        name=["sort"],
        dtype=__Skip(np.float16),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((11400, ),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),
    'sub_scalar': dict(
        name=['sub'],
        # dtype=[np.float32],
        # para=dict(
        #     other=[0, -1, 0.028, 2.232, 1, -0.2421, -2],
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (1024,), (128, 64, 3, 3), (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'sub_constant_with_alpha_and_no_contiguous': dict(
        name=['sub'],
        # para=dict(
        #     alpha=[0, -2, 2.0, 4, 1, 0.234, -2.123],
        #     other=[3.5, -2, 2.0, 4, 1, -0.231, 3],
        # ),
        # dtype=[np.float32],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (1024, ), (384, 128), (2, 64, 128),
                             (128, 64, 3, 3), (128, 32, 2, 2),
                             (2, 32, 130, 130)),
                },
            ],
        ),
    ),

    'matmul': dict(
        name=["matmul"],
        rtol=1e-5,
        atol=8e-3,
        dtype=__Skip(np.float16, np.float64, np.int64),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape":__Skip((5,), (2, 1, 3136, 3136), (2, 16, 8, 64), (2, 31, 6, 40, 512))
                },
            ],
        ),
    ),
    'reduce_op': dict(
        name=['mean', 'sum'],
        atol=1e-4,
        rtol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((),(64, ), (169, 4), (17100, 2), (1, 1, 384),
                               (4, 133, 128, 128), (2, 64, 3, 3, 3),
                              (0,), (0, 2), (16, 0, 9)),
                    "dtype": __Skip(np.float64, np.float16),
                },
            ],
        ),
    ),
    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        # para=dict(
        #     dim=[-1, 0, 1, [0, 1], 2, [-1, 0, 2], 3,
        #          [0], -2, [0, 1]],
        # ),
        atol=1e-4,
        rtol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (64, ), (17100, 2), (2, 64, 3, 3, 3),
                              (0,), (0, 2), (16, 0, 9), (4, 133, 128, 128)),
                    "dtype": __Skip(np.float64, np.float16),
                },
            ],
        ),
    ),
    'reduce_partial_op_4': dict(
        name=['sum'],
        # para=dict(
        #     dim=__Skip(-1, 0, 1, [0, 1], 2, [-1, 0, 2], 3,
        #          [0], -2, [0, 1]),
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((),
                    #(64, ), (169, 4), (17100, 2), (1, 1, 384),
                    #           (4, 133, 128, 128), (2, 64, 3, 3, 3),
                    #           (0,), (0, 2), (16, 0, 9)
                    ),
                    "dtype": __Skip(np.int16, np.int32, np.int64,
                              np.uint8, np.int8, np.bool_),
                },
            ],
        ),
    ),
    'transpose': dict(
        name=['transpose'],
        dtype=__Skip(np.float16, np.float64, np.int16,
                np.int64, np.uint8, np.int8, np.bool_),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (0, 8), (16, 0, 8)),
                },
            ],
        ),
    ),

    'triu': dict(
        name=['triu'],
        # para=dict(
        #     diagonal=[0, 1, 2, -1, 3, 12, 0, 5, -9, -1, 1, 2, 10, -10],
        # ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((9, 9), (2, 0), (12, 0), (2, 0, 9)),
                    "dtype": __Skip(np.float16, np.int16, np.int32, np.int64,
                               np.uint8, np.int8, np.bool_),
                },
            ],
        ),
    ),

    'multinomial': dict(
        name=["multinomial"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((8, ), (8, ), (8, ),
                    #           (16, 64,), (128, 256,), (256, 128,),
                              (0, 8), (0, 8)),
                    "dtype": __Skip(np.float16, np.float64),
                },
            ],
        ),
    ),

    'silu': dict(
        name=["silu"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((0,), (0, 16), (8, 0, 17)),
                    "dtype": __Skip(np.float16, np.float64),
                },
            ],
        ),
    ),
    'uniform': dict(
        name=['uniform'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": __Skip((), (0,), (4, 0), (3, 0, 9)),
                    "dtype": __Skip(np.float64, np.float16),
                },
            ],
        ),
    ),
    'where': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": __Skip((),(1024,), (1482,4), (0,), (2, 0), (2, 0, 9)),
                    "dtype": __Skip(np.uint8)
                },
                {
                    "ins": ['input', 'other'],
                    "dtype": __Skip(np.float16, np.float64, np.int16,
                              np.int64, np.uint8, np.int8, np.bool_),
                    "shape": __Skip((), (0,), (2, 0), (2, 0, 9)),
                },
            ],
        ),
    ),

    'where_broadcast': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": __Skip((), (0,), (2, 0), (2, 0, 9)),
                    "dtype": __Skip(np.uint8)
                },
                {
                    "ins": ['input'],
                    "dtype": __Skip(np.float64),
                },
                {
                    "ins": ['other'],
                    "dtype": __Skip(np.float64),
                    "shape": __Skip((0,), (2, 1), (0, 1)),
                },
            ],
        ),
    ),
    'where_same_value': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": __Skip((1, 445), (3, 5), (4,), (3,4,5), (3,)),
                },
                {
                    "ins": ['input'],
                    "dtype": __Skip(np.float64),
                },
                {
                    "ins": ['other'],
                    "dtype": __Skip(np.float64),
                },
            ],
        ),
    ),
}
