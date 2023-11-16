# Copyright (c) 2023, DeepLink.

# Note : 1. aten's cuda implementation doesn't support 3-dim nhwc Tensor
#        adaptive_max_pool2d（3d), max_pool3d,
#        adaptive_avg_pool3d, interpolate doesn't support nhwc memory format
#        avg_pool2d backward can't compute right along the edges
#        2. For camb test, adaptive_max_pool2d/max_pool2d need indices being int32
#        Only conv2d, bn, adaptive_avg_pool2d, adaptive_max_pool2d can be tested, because
#        the rest have't been implemented.
nhwc_op = {'conv2d': ["2d", "input", 'weight'],
           'conv3d': ["3d", "input", 'weight'],
           'batch_norm': ['input'],
           'adaptive_avg_pool2d': ["2d", 'input'],
           'adaptive_max_pool2d': ["2d", 'input'],
           'adaptive_avg_pool3d': ["3d", 'input'],
           'adaptive_max_pool3d': ["3d", 'input'],
           'avg_pool2d': ["2d", 'input'],
           'max_pool2d': ["2d", 'input'],
           # 'avg_pool3d': ["3d", 'input'], diopi doesn't hava avg_pool3d test
           'max_pool3d': ["3d", 'input'],
           # both embedding
           'interpolate': ['input'],
           'pad': ['input'],
           'roi_align': ['input']}

# Note : 1. camb test: all ops implemented is passed.
#        2. nv test: most of ops is not implemented for 'Int'.
#           Tests of index_select, bce, embedding passed for 'Int'.
dtype_op = {'nll_loss': ['target'],  # input using int32/float32 type
            'cross_entropy': ['target'],
            'index_select': ['index'],
            'index_put': ['indices1', 'indices2'],
            'binary_cross_entropy_with_logits': ['pos_weight'],
            'gather': ['index'],
            'scatter': ['index'],
            'embedding': ['input'],
            'index': ['idx1', 'idx2'],
            'ctc_loss': ['targets', 'input_lengths', 'target_lengths'],
            'index_fill': ['index'],
            'one_hot': ['input']}

# Note : 1. camb test: all ops implemented is passed.
#        2. nv test: most of ops is not implemented for 'Int'.
#           Tests of unique, arange, randperm, argmax passed for 'Int'.
dtype_out_op = {'max_pool2d': ['indices'],  # out using int32/float32 type
                'max_pool3d': ['indices'],
                'adaptive_max_pool2d': ['indices'],
                'adaptive_max_pool3d': ['indices'],
                'max': ['indices'],
                'min': ['indices'],
                'sort': ['indices'],
                'topk': ['indices'],
                'unique': ['indices'],
                'one_hot': ['out'],
                'arange': ['out'],
                'randperm': ['out'],
                'argmax': ['out']}

ops_with_states = {"batch_norm": {"running_mean", "running_var"},
                   "sgd": {"buf", "param"},
                   "fill_": {"input"},
                   "embedding": {"weight"},
                   "adam": {"param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
                   "adamw": {"param", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"},
                   "adadelta": {"param", "square_avg", "acc_delta"},
                   "rmsprop": {"param", "square_avg", "grad_avg", "momentum_buffer"},
                   "copy_": {"input"},
                   "cast_dtype": {"out"},
                   "batch_norm_gather_stats_with_counts": {"running_mean", "running_var"},
                   "apply_penalty": {"logits"},
                   "context_attention": {"out"},
                   "destindex_copy_kv": {"out"},
                   "token_attention": {"out"},
                   "token_softmax_reducev": {"out"}
                   }
