model_list = ['ResNet50', 'VGG16']

model_op_list = {
    'ResNet50': ['sgd', 'random', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'addmm', 'cross_entropy', 'log_softmax', 'nll_loss', 'sum', 'mean'],
    'VGG16': ['sgd', 'random', 'randperm', 'conv2d', 'relu', 'batch_norm', 'max_pool2d', 'linear', 'addmm', 'dropout', 'cross_entropy', 'log_softmax', 'nll_loss', 'sum', 'mean', 'add'],
}
