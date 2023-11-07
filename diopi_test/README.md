<div align=center>
<img src="../img/deepLink_logo.png">
</div>

# DIOPI_TEST

DIOPI_TEST是构建于设备无关算子接口（Device-Independent Operator Interface, DIOPI）之上的测试框架，它支持了没有训练框架的情况下，验证算子适配正确性的功能。DIOPI_TEST设计了一套完整的测试框架和一套算子函数测试。测试套件，可以使芯片厂商适配 DIOPI 算子时，无需训练框架即可对适配结果的正确性进行验证。

主要模块：
* DIOPI_TEST 运行时：支持了运行时函数的接口，用以管理设备相关资源。
* 非算子测试：
    * 测试获取设备相关信息标准接口。
    * 测试获取错误信息标准接口。
    * 测试上下文 Context 中 Stream 的正确使用。
* 算子测试：
    * 自定义测例配置：套件提供了描述算子测例的配置文件，用户可自定义扩展测例。
    * 生成基准数据：套件可以根据配置文件生成算子测例的基准输入和输出数据。
    * 校验适配算子：算子适配完成后使用基准输入数据得到的结果与基准输出数据进行比较验证。
* 模型算子测试：
    * 采用算子测试相同的测例配置规则, 使用同一个测试框架生成基准数据并进行测试验证。
    * 从40多个模型训练过程中抓取张量形状，数据类型及其他非张量参数值生成测例。


DIOPI_TEST 测试范围：
* 每一个 DIOPI 标准算子均有相应的测试，并且会从不同的数据类型、张量维度、非张量参数等角度对每个算子设计多个测例。保证 DIOPI 标准算子接口中每个参数功能均被测试。针对常见训练算子目前已有约 11000+个测例， 其中涵盖了如 conv2d， batch_norm, adaptive_max_pool2d, relu 等经典训练算子。
* DIOPI_TEST 提供的模型算子测试，涵盖了经典分类模型如 resnet50, vgg16, seresnet50, densenet, mobilenet_v2, efficientnet, shufflenet_v2, repvgg, swin_transformer, vit, inceptionv3 及经典检测模型如 retinanet, faster_rcnn_r50, ssd300, yolov3, atss, fcos, mask_rcnn, solo, centernet, cascade_rcnn, detr 及经典分割模型如 unet, upernet, pspnet, fcn, deeplabv3, deeplabv3plus 及其他领域深度学习模型 sar, dbnet, stgcn, crnn, hrnet, deeppose, tsn, slowfast。


## **使用教学**

参考 [DIOPI Readme](https://github.com/DeepLink-org/DIOPI/#readme)

### 结果分析

测例通过的输出形式如下：
```
collecting ... collected 1 items

gencases/diopi_case/test_diopi_add_add.py::TestMdiopiSaddFadd::test_add_0 PASSED [100%]
```
如需输出HTML格式报告：
```
pip install pytest-testreport
python main.py --mode run_test --html_report
```

### 可选测试模式
DIOPI_TEST框架还提供针对不同硬件芯片特点的测试模式以及其他测试模式

* mode: 指定测试阶段

    mode 为 gen_data 时产生基准输入输出数据：
    ```
    python main.py --mode gen_data
    ```
    mode 为 gen_case 时产生测例文件：
    ```
    python main.py --mode gen_case
    ```
    mode 为 run_test 时运行测试：
    ```
    python main.py --mode run_test
    ```
    mode 为 utest 时运行非算子测试：
    ```
    python main.py --mode utest
    ```
* fname: 指定算子测试

    fname 为函数名字选项, 如果指定函数名字 (测例配置文件中测例的 name) 则会对该算子进行基准数据生成和测试,
    不指定默认对所有算子生成基准数据和测试。fname 默认值为 all_ops。

    ```
        # 只测试 relu
        python main.py --mode gen_data --fname relu
        python main.py --mode gen_case --fname relu
        python main.py --mode run_test --file_or_dir /path/to/relu/case

        # 测试所有算子
        python main.py --mode gen_data
        python main.py --mode gen_case
        python main.py --mode run_test

        # 测试所有算子
        python main.py --mode gen_data --fname all_ops
        python main.py --mode gen_case --fname all_ops
        python main.py --mode run_test
    ```

* filter_dtype: 过滤指定数据类型的测试

    当前测试方案中, 会在配置文件中配置算子支持的多个数据类型, 比如: int32, int64, float32, float64。
    默认的测试行为会对所有配置的数据类型都进行测试，但是可能存在某些硬件并不支持所有配置的数据类型，
    比如不支持 float64, 那么可以通过设置 filter_dtype 为 float64 来过滤掉对于 float64 的测试。

    ```
        python main.py --mode run_test --file_or_dir /path/to/case --filter_dtype float64
        # 可叠加不支持的数据类型
        python main.py --mode run_test --file_or_dir /path/to/case --filter_dtype float64 int64
    ```

* nhwc : 使用 channel_last 格式的张量测试

    目前，模型中使用到的数据格式主要为 nchw/nhwc 和 ncdhw/ndhwc。当前测试默认支持的是 nchw/ncdhw 数据格式。
    如果需要测试 nhwc/ndhwc 格式，可以通过设置 nhwc 来生效。

    channel_last 测试只对部分算子有效, 请参考 python/conformance/utils.py 中 nhwc_op 字典。
    其中, key 为需要使用 channel last 数据格式的算子名称, value 的第一个参数表示 2d/3d 数据。
    如果没有显式指明，如 interpolate 算子, 则对 4 维以下的张量按照 2d 数据处理, 5 维张量按照
    3d 数据处理, 目前不支持 5 维以上输入。value 后续元素代表算子需要转换为 channel last 数据格式的
    参数。

    出于统一管理基准输入输出数据的目的, 且数据格式是否为 channel last 并不影响最终计算结果。
    故数据格式的转换仅发生在 run_test 阶段。

    ```
        # --nhwc 仅对在 nhwc_op 字典中的算子有效
        python main.py --mode gen_case --fname relu --nhwc
        python main.py --mode run_test --file_or_dir /path/to/relu/case
    ```


* four_bytes: 使用int32代替int64测试

    pytorch 需要索引张量的算子 (max_pool, sort等), 基本采用 int64 作为默认数据格式。
    而很多国产 AI 芯片并不支持该数据类型运算, 在底层核函数中使用 int32 数据类型代替 int64 计算。
    为了支持国产 AI 芯片这一特性, 一致性测试框架允许使用 int32 数据类型进行测试。

    该设置只对部分算子有效, 请参考 python/conformance/utils.py 中 dtype_op 字典。
    其中, key 为使用 int32 代替 int64 的算子名称, value 中为使用 int32
    数据类型的输入变量或输出变量。

    出于统一管理基准输入输出数据的目的, 且均是整型数据类型, 对于精度不会产生明显影响。
    故数据类型的转换仅发生在 run_test 阶段。

    ```
        # --four_bytes 仅对在 dtype_op/dtype_out_op 字典中的算子有效
        python main.py --mode gen_case --fname relu --four_bytes
        python main.py --mode run_test --file_or_dir /path/to/relu/case
    ```

* model_name: 指定模型相关算子测试

    为了简化模型相关的算子测试，可以通过设置 model_name 来测试指定模型的所有算子。

    ```
        python main.py --mode gen_data --model_name resnet50
        python main.py --mode gen_case --model_name resnet50
        python main.py --mode run_test --file_or_dir /path/to/resnet50/case
    ```
### 测例配置说明

DIOPI-TEST 设计了一套测例配置规则及相应的测试框架。以算子测试为例，所有算子测例配置文件位于 python/conformance/diopi_configs.py 中。
我们以 group_norm 算子测例配置为例来阐释说明测例生成。

```
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
```
* name: *list*
    函数名字。

    如在测试 DIOPI 算子时使用到的 python 函数名字：

    ```
    diopi_funtions.group_norm(**kwargs)
    ```

* interface: *list*

    在生成基准数据中使用模块名, 若不指定, 默认为 torch.nn.functional。
    如在生成基准数据时调用的 pytorch 函数:

    ```
    torch.group_norm(**kwargs)
    ```

* atol: *float*
    用于结果验证的绝对误差参数。

* rtol: *float*
    用于结果验证的相对误差参数。
    结果验证使用此函数检查输出 out  和参考输出 out_ref 是否满足条件：

        |out - out_ref| ≤ atol + rtol x |out_ref|

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
    - gen_policy (*str*): 生成张量的策略, 默认为'default'

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
        用于 half 数据类型测试结果验证的绝对误差参数。
    * rtol_half: *float*
        用于 half 数据类型测试结果验证的相对误差参数。
    * is_inplace: *bool*
        是否复用基准输入输出数据做 inplace 版本算子测试。默认为 Fasle。
    * no_out_ref: *bool*
        常见于随机数算子测试中，用以表明该算子测试无基准输出数据。
    * saved_args: *dict*
        指定输出结果作为反向计算的输入参数。

### 厂商自定义测例配置
我们提供了厂商自定义测例配置的能力，可以对python/conformance/diopi_configs.py里的测例按条件进行跳过，以及修改误差参数。

如果要进行自定义配置，需要创建名为device_configs.py的配置文件，在文件里添加

```
    from .device_config_helper import Skip
    from .diopi_runtime import Dtype
    device_configs = {}
```


并可以按以下方式在device_configs里进行配置：

1. 修改误差参数

```
    'cdist': dict(
        name=['cdist'],
        atol=1e-06,
        rtol=1e-05,
    ),
```

以上配置会覆盖原有配置中的atol, rtol。类似的，我们也支持atol_half，rtol_half的修改。

2. 跳过任意tensor_para为某个类型的测例

```
    'cdist': dict(
        name=['cdist'],
        dtype = [Skip(Dtype.float64)],
    ),
```

原本的python/conformance/diopi_configs.py中，对应的配置有x1, x2两个tensor_para。以上配置会跳过所有x1为float64或x2为float64的测例。

3. 跳过特定参数条件的测例

```
    'cdist': dict(
        name=['cdist'],
        para=dict(
            p=[Skip(1),],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ],
        ),
    ),
```

以上配置会跳过p=1 __或__ x1为float64类型的测例。


一个完整的device_configs.py的示例如下:
```
    from .device_config_helper import Skip
    from .diopi_runtime import Dtype
    device_configs = {
        'cdist': dict(
            name=['cdist'],
            para=dict(
                p=[Skip(1),],
            ),
            tensor_para=dict(
                args=[
                    {
                        "ins": ['x1'],
                        "dtype": [Skip(Dtype.float64)],
                    },
                ],
            ),
        ),
    }
```


创建完device_configs.py后，可以在验证算子时通过impl_folder参数将device_configs.py所在文件夹路径传进来，


```
    python main.py --mode gen_case --impl_folder /path/to/folder --fname cdist
    python main.py --mode run_test --path_or_dir /path/to/cdist/case
```

