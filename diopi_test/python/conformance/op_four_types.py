# Copyright (c) 2023, DeepLink.

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
