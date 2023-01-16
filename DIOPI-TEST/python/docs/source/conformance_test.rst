DIOPI 测例说明
===================

配置文件规则
------------------------

所有算子测例配置文件位于 python/conformance/diopi_configs.py 中。
以 group_norm 算子测例配置为例来阐释说明测例生成。

.. code-block:: python

    'group_norm': dict(
        name=['group_norm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            num_groups=[32, 32, 32, 32],
            eps=[1e-05, 1e-05, 1e-05, 1e-05]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256, 100, 152), (2, 256, 7, 10),
                              (2, 256, 24, 24), (2, 256, 12, 12)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((256,), (256,),
                              (256,), (256,)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

* name: *list*
    函数名字。

    如在测试 DIOPI 算子时使用到的 python 函数名字：
    
    .. code-block:: python

        diopi_funtions.group_norm(**kwargs) 

* interface: *list*

    在生成基准数据中使用模块名, 若不指定, 默认为 torch.nn.functional。
    如在生成基准数据时调用的 pytorch 函数:

    .. code-block:: python
    
        torch.group_norm(**kwargs)

* atol: *float*
    用于结果验证的绝对误差参数, 默认值为 1e-8。

* rtol: *float*
    用于结果验证的相对误差参数, 默认值为 1e-5。
    结果验证使用此函数检查输出 out  和参考输出 out_ref 是否满足条件：
    \|out - out_ref\| ≤ atol + rtol x \|out_ref\|

* para: *dict*
    函数非张量参数（比如 num_groups, eps 等）。

* tensor_para: *dict*
    函数张量参数（比如 input, weight, bias 等）。

    args 中：

        - ins (*list*): 张量参数的名字, 默认为 ["input"]。
        - shape (*tuple*): 张量参数的形状。
        - gen_fn (*builtin_function*): 数据生成器, 默认为 numpy.random.randn。
          若在 args 外部制定, 则应用于 args 内所有张量参数。
        - requires_grad (*list*): 是否反向计算梯度, 默认为 [Fasle]。
        - dtype (*list*): 张量的数据类型, 若在 args 外部制定, 则应用于 args 内所有张量参数。

    args 中包含 **多组** 测试用例张量, shape 元素个数代表张量组数, 每个 ins 的 shape 都是 **一一对应** 的。
    该数量于 **para** 每个非张量参数的数量也 **一一对应**。
    group_norm 中描述了四组输入张量测试用例, 以下四组测例分别为不同数据类型，此测例配置为 float32, float64
    各生成一次基准输入输出数据:

    第一组: `group0 = {"num_group": 32, "eps" : 1e-05, "input": tensor(2, 256, 100, 152), "weight": tensor(256), "bias": tensor(256)}`
    
    第二组: `group1 = {"num_group": 32, "eps" : 1e-05, "input": tensor(2, 256, 7, 10), "weight": tensor(256), "bias": tensor(256)}`
    
    第三组: `group2 = {"num_group": 32, "eps" : 1e-05, "input": tensor(2, 256, 24, 24), "weight": tensor(256), "bias": tensor(256)}`

    第四组: `group3 = {"num_group": 32, "eps" : 1e-05, "input": tensor(2, 256, 12, 12), "weight": tensor(256), "bias": tensor(256)}`

* 其他配置参数：
    * atol_half: *float*
        用于 half 数据类型测试结果验证的绝对误差参数, 默认值为 1e-4。
    * rtol_half: *float*
        用于 half 数据类型测试结果验证的相对误差参数, 默认值为 5e-3。
    * is_inplace: *bool*
        是否复用基准输入输出数据做 inplace 版本算子测试。默认为 Fasle。
    * no_out_ref: *bool*
        常见于随机数算子测试中，用以表明该算子测试无基准输出数据。
    * saved_args: *dict*
        指定输出结果作为反向计算的输入参数。具体细节见 :ref:`反向测试说明 <反向测试基准数据生成说明>`。
    * seq_name: *str* 和 gen_num_range : *list*
        见于cat、stack算子的测例配置, 组合使用。gen_num_range 表示在指定的范围内产生随机数个 args 中的张量。
        seq_name 指示将这些放入列表中的张量列表名字。


可选测试模式
------------------------
* fname: 指定算子测试
    fname 为函数名字选项, 如果指定函数名字 (测例配置文件中测例的 name) 则会对该算子进行基准数据生成和测试,
    不指定默认对所有算子生成基准数据和测试。fname 默认值为 all。

    .. code-block:: shell

        # 只测试 relu
        python main.py --mode gen_data --fname relu
        python main.py --mode run_test --fname relu

        # 测试所有算子
        python main.py --mode gen_data
        python main.py --mode run_test

        # 测试所有算子
        python main.py --mode gen_data --fname all
        python main.py --mode run_test --fname all


* filter_dtype: 过滤指定数据类型的测试
    当前测试方案中, 会在配置文件中配置算子支持的多个数据类型, 比如: int32, int64, float32, float64。
    默认的测试行为会对所有配置的数据类型都进行测试，但是可能存在某些硬件并不支持所有配置的数据类型，
    比如不支持 float64, 那么可以通过设置 filter_dtype 为 float64 来过滤掉对于 float64 的测试。

    .. code-block:: shell

        python main.py --mode gen_data --fname relu --filter_dtype float64
        python main.py --mode run_test --fname relu --filter_dtype float64

        # 可叠加不支持的数据类型
        python main.py --mode run_test --fname relu --filter_dtype float64  --filter_dtype int64

* nhwc : 使用 channel_last 格式的张量测试
    目前，模型中使用到的数据格式主要为 nchw/nhwc 和 ncdhw/ndhwc。当前测试默认支持的是 nchw/ncdhw 数据格式。
    如果需要测试 nhwc/ndhwc 格式，可以通过设置 nhwc 来生效。

    channel_last 测试只对部分算子有效, 请参考 python/conformance/utils.py 中 nhwc_op 字典。
    其中, key 为需要使用 channel last 数据格式的算子名称, value 的第一个参数表示 2d/3d 数据。
    如果没有显式指明，如 interpolate 算子, 则对 4 维以下的张量按照 2d 数据处理, 5 维张量按照
    3d 数据处理, 目前不支持 5 维以上输入。value 后续元素代表算子需要转换为 channel last 数据格式的
    参数。

    .. code-block:: python

        nhwc_op = { 'conv2d':["2d", "input", 'weight'],
                    'conv3d':["3d", "input", 'weight'],
                    'batch_norm':['input'],
                    'adaptive_avg_pool2d':["2d", 'input'],
                    'adaptive_max_pool2d':["2d", 'input'],
                    'adaptive_avg_pool3d':["3d", 'input'],
                    'adaptive_max_pool3d':["3d", 'input'],
                    'avg_pool2d':["2d", 'input'],
                    'max_pool2d':["2d", 'input'], 
                    'max_pool3d':["3d", 'input'], 
                    'interpolate':['input'],
                    'pad':['input'],
                    'roi_align':['input']
                  }

    出于统一管理基准输入输出数据的目的, 且数据格式是否为 channel last 并不影响最终计算结果。
    故数据格式的转换仅发生在 run_test 阶段。
    
    .. code-block:: shell

        # --nhwc 仅对在 nhwc_op 字典中的算子有效
        python main.py --mode run_test --fname relu --nhwc
    


* four_bytes: 使用int32代替int64测试
    pytorch 需要索引张量的算子 (max_pool, sort等), 基本采用 int64 作为默认数据格式。
    而很多国产 AI 芯片并不支持该数据类型运算, 在底层核函数中使用 int32 数据类型代替 int64 计算。
    为了支持国产 AI 芯片这一特性, 一致性测试框架允许使用 int32 数据类型进行测试。

    该设置只对部分算子有效, 请参考 python/conformance/utils.py 中 dtype_op 字典。
    其中, key 为使用 int32 代替 int64 的算子名称, value 中为使用 int32 
    数据类型的输入变量或输出变量。

    .. code-block:: python

        dtype_op = { # 输入使用 int32 的算子及变量名
                    'nll_loss' : ['target'],
                    'cross_entropy' : ['target'],
                    'index_select' : ['index'],
                    'index_put' : ['indices1', 'indices2'],
                    'binary_cross_entropy_with_logits' : ['pos_weight'],
                    'gather' : ['index'],
                    'scatter' : ['index'],
                    'embedding' : ['input'],
                    'index' : ['idx1', 'idx2'],
                    'ctc_loss' : ['targets', 'input_lengths', 'target_lengths'],
                    'index_fill' : ['index'],
                    'one_hot' : ['input'],
                }
    
        dtype_out_op = { # 输出使用 int32 的算子及变量名
                'max_pool2d' : ['indices'], 
                'max_pool3d' : ['indices'],
                'adaptive_max_pool2d' : ['indices'],
                'adaptive_max_pool3d' : ['indices'],
                'max' : ['indices'],
                'min' : ['indices'],
                'sort' : ['indices'],
                'topk' : ['indices'],
                'unique' : ['indices'],
                'one_hot' : ['out'],
                'arange' : ['out'],
                'randperm' : ['out'],
                'argmax' : ['out']
            }

    出于统一管理基准输入输出数据的目的, 且均是整型数据类型, 对于精度不会产生明显影响。
    故数据类型的转换仅发生在 run_test 阶段。

    .. code-block:: shell

        # --four_bytes 仅对在 dtype_op/dtype_out_op 字典中的算子有效
        python main.py --mode run_test --fname relu --four_bytes

* model_name: 指定模型相关算子测试
    为了简化模型相关的算子测试，可以通过设置 model_name 来测试指定模型的所有算子。
    该设置会屏蔽对于 fname 的制定。

    .. code-block:: shell

        python main.py --mode gen_data --model_name ResNet50
        python main.py --mode run_test --model_name ResNet50

反向测试规则
------------------------

1. 反向测试算子范围
~~~~~~~~~~~~~~~~~~~~~~~~

      并非所有算子都要进行反向测试, 这是因为部分 DIOPI 算子并没有对应的反向算子声明。譬如 DiopiAdd, 
      因为 add 算子的反向也通过 add 实现, 故没有必要声明一个类似于 DiopiAddBackward 算子。
      常见训练框架实现自动微分时, 也一般是复用 add 算子计算反向。另外, 即使训练框架真的定义了一个叫 add_backward
      的函数, 在框架适配 DIOPI 算子时，我们也只需要将 DiopiAdd 包装进 add_backward 即可。

      具体哪些算子需要测试反向, 可以通过 diopirt/include/diopi/functions.h 中函数声明查询, 若存在 DIOPI 反向算子声明,
      则该算子一定会有相应的反向测试。
      另外, 也可以查询 python/conformance/diopi_configs.py 文件, 若对 tensor_para 中 args 之一的张量将其 requires_grad
      属性设置为 True, 则该算子会同时测试其相应的反向算子。以 log_softmax 的测例配置为例： 

      .. code-block:: python

        'log_softmax': dict(
            name=["log_softmax"],
            saved_args=dict(output=0), # 指定反向算子需要的第 x 个前向输出结果
            para=dict(
                dim=[-1, 1, 0],
            ),
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "requires_grad": [True], # requires_grad 为 True 则需要反向测试
                        "shape": ((78, 24), (2, 92, 29), (2, 150, 512, 512)),
                        "dtype": [Dtype.float32, Dtype.float64],
                        "gen_fn": Genfunc.randn,
                    },
                ],
            ),
        ),

.. _反向测试基准数据生成说明:

2. 反向测试基准数据
~~~~~~~~~~~~~~~~~~~~~~~~

    * **反向测试的基准输出数据** 使用 pytorch 的 torch.autograd 接口自动在每个前向算子完成时进行反向计算,
      并将计算结果保存下来作为基准输出数据。另外, 初始回传梯度通过 ones_like 生成, 初始梯度与输出大小相同但值均为 1。
      如果前向算子有多个输出, 可以通过 diopi_configs.py 配置文件中的 requires_backward 的值, 如指定
      requires_backward=[0], 则只对第 1 个输出结果张量创建梯度并回传。目前暂无算子测例使用 requires_backward 属性。

    .. code-block:: python

        class GenOutputData(object):
            r'''
            Generate output data for all functions by using torch and input data
            '''
            @staticmethod
            def run(func_name, model_name, filter_dtype_str_list):
                ...
                for saved_pth in saved_pth_list: # 循环每个算子测例
                    ...
                    if function_paras["requires_grad"]: # 判断是否需要反向测试
                        ...
                        # 若未指定 requires_backward 则对所有前向输出结果张量创建梯度
                        # 否则, 仅对指定前向输出结果张量创建梯度
                        requires_backward = data["cfg"]["requires_backward"]
                        outputs_for_backward = outputs if len(requires_backward) == 0 \
                        else [outputs[i] for i in requires_backward]

                        inputs_name_for_grad, inputs_for_grad = get_name_and_data_for_grad(function_paras)
                        saved_grads = None
                        if len(inputs_for_grad) != 0:
                            # 通过 ones_like 函数创建初始梯度
                            grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                            # 通过 torch.autograd.grad 自动微分进行反向计算, 得到反向基准输出数据
                            grads = torch.autograd.grad(
                                outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                            saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}

    * **反向测试的基准输入数据** 主要是复用前向的输入参数和以及指定的输出结果。以上述 log_softmax 为例, 在调用 python 层反向算子时, 会将所有的前向参数
      dim, input 传入 python 层反向算子。 另外如果指定了 saved_args, 还需要传递 saved_args 指定的前向输出结果。如 log_softmax 测例指定了
      saved_args=dict(output=0), 且 log_softmax 只返回一个输出, 故这里会将第一个输出也是唯一的输出传递给反向算子。

      另外有些前向参数可能不被反向计算所需要, 这里是通过 \*\*kwargs 不指定关键字参数个数来处理。这是因为一致性测试框架主要以键值对的方式传参到 
      python/conformance/diopi_functions.py 中的 python 函数接口进行测试。我们在定义反向函数接口时,
      会添加一个 \*\*kwargs 参数来接受不被使用的关键字参数。

      .. code-block:: python

        def log_softmax(input, dim, dtype=None):
            ...

        # 所有 python 层反向算子接口均以前向函数名加上 _backward 命名
        # 所有 python 层反向算子接口均有 **kwargs 参数以接受不定长且不被使用的前向算子参数
        def log_softmax_backward(input, grad_outputs, output, dim, **kwargs):
            ...

3. 反向测试运行机制
~~~~~~~~~~~~~~~~~~~~~~~~

    - 在 diopi_configs.py 配置文件中为有反向声明的 DIOPI 算子通过指定输入张量的 requires_grad
      属性为 True 来表示需要进行反向测试

    - 反向测试打包所有前向参数以及 saved_args 中指定的某个前向输出结果到 python 反向函数接口。
      在 diopi_functions.py 封装的函数中, 反向函数以前向函数名加上 _backward 命名,
      另外添加 \*\*kwargs 来接受不定长的关键字参数。传参逻辑如下：

    .. code-block:: python

        class ConformanceTest(object):
            r'''
            Run all functions by using input, then compare_with_gen_output with saved output
            '''
            @staticmethod
            def run(func_name, model_name, filter_dtype_str_list):
                ...
                for saved_pth in saved_pth_list: # 循环每个算子测例
                    ...
                    # 判断是否需要反向测试
                    if function_paras["requires_grad"] and "inplace=True" not in func_call:
                        ...
                        # requires_backward 作用同上，用以创建指定输出张量的梯度
                        requires_backward = data["cfg"]["requires_backward"]
                        outputs_for_backward = output if len(requires_backward) == 0 \
                            else [output[i] for i in requires_backward]

                        backward_para = {}
                        grad_outputs = [F.ones_like(i) for i in outputs_for_backward]
                        backward_para["grad_outputs"] = grad_outputs
                        # 将 saved_args 中指定的前向输出存在 backward_para 字典中
                        for k, v in data["cfg"]['saved_args'].items():
                            backward_para[k] = output[v]

                        try:
                            # 将所有前向算子的关键字参数以及 backward_para 打包传递给反向算子
                            grad_input = eval(f"F.{cfg_func_name}_backward(**kwargs, **backward_para)")
                        ...
        
    - 在一致性测试框架中计算反向结果, 并同基准输出数据对比。