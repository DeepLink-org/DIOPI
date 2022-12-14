# ConformanceTest-DIOPI

<!-- 简体中文｜[English](README.md) -->

## 简介
一致性测试套件（ConformanceTest-DIOPI）是构建于[设备无关算子接口（Device-Independent Operator Interface, DIOPI）](https://github.com/OpenComputeLab/DIOPI)之上的测试框架，它支持了没有训练框架的情况下，验证算子适配正确性的功能。

主要模块：
- [DIOPI 运行时](csrc)：支持了运行时函数的接口，驱动芯片对算子进行运算。
- DIOPI 算子实现：提供了接入一致性测试套件的算子开发示例。
    + [impl/cuda](impl/cuda)：使用 CUDA 和 cuDNN 实现了小部分英伟达算子接口。
    + [impl/torch](impl/torch)：使用 PyTorch C++ API 实现了英伟达算子接口。
    + [impl/camb](impl/camb)：使用 CNNL 实现了小部分寒武纪算子接口。
    + [impl/camb_pytorch](impl/camb_pytorch)：使用 camb_pytorch 实现了寒武纪算子接口。
- [算子测试](python/main.py)：
    + 自定义测例配置：套件提供了描述算子测例的配置文件，方便用户自定义扩展测例。
    + 生成基准数据：套件可以根据配置文件生成算子测例的基准输入和输出数据。
    + 校验适配算子：算子适配完成后使用基准输入数据得到的结果与基准输出数据进行比较验证。

## <span id="start">开始</span>
以 [impl/cuda](impl/cuda) 为例介绍如何使用一致性测试套件进行算子测试，使用之前请确保 CUDA 和 PyTorch 已经成功安装在环境中，生成基准数据和运行 impl/cuda 算子测例要求 `cuda>=10.1`， `pytorch>=1.7.0`。


### i. 准备 DIOPI 子模块
```bash
git submodule update --init
```

### ii. 编译
编译后可以在 lib 目录查看到生成的 `libdiopirt.so` 和 `libdevice_impl.so` 动态库。
```bash
mkdir -p build && cd build

cmake .. -DCUDA_ARCH_AUTO=ON -DIMPL_OPT=Torch

make -j4
```
同时项目中提供了编译脚本 `sh scripts/build_impl.sh torch`，可以直接运行进行编译。


### iii. 测试
第一步生成基准输入和输出数据，第二步验证适配的算子的正确性。
</br>测试脚本运行命令：
</br>`python main.py [-h] [--mode MODE] [--fname FNAME]`

选项说明：

- --mode *可选项：gen_data, run_test, utest*
</br> 运行模式选项，用于选择当前函数生成基准数据还是测试算子

- --fname *缺省：all*
</br> 函数名字选项，如果指定函数名字（配置文件中测例的 *name*）则会对该算子进行基准数据生成和测试，不指定默认对所有算子生成基准数据和测试。

```bash
cd ../python

# Step 1: 在 Nvidia 设备上生成基准输入和输出数据
python main.py --mode gen_data --fname all

# Step 2: 在接入芯片设备上运行测试
python main.py --mode run_test --fname all

# Other: 用于验证接入芯片是否支持框架 diopiTensor 的基本操作
python main.py --mode utest
```
## 其他定制化测试
</br>测试脚本运行命令：
</br>`python main.py [-h] [--get_model_list]`

选项说明：
</br> 执行该模式, 将返回目前已有的模型及模型所需算子清单

</br>测试脚本运行命令：
</br>`python main.py [-h] [--mode MODE] [--model_name NAME] [--filter_dtype TYPE]`

选项说明：

- --mode_name *可选项：选择的模型名字, 指定后 --fname 将失效
</br> 指定模型选项，当前 mode 对模型所需算子清单中的所有算子执行
- --filter_dtype *可选项：过滤的数据类型
</br> 指定过滤的数据类型选项，当前 mode 对指定的数据类型不执行

## 配置文件规则
配置文件 [diopi-conifg.py](python/conformance/diopi_configs.py) 使用 Python 语法以字典 key-value 的形式描述算子测例，key 为测例的名字，value 包含了测例的参数配置。下面以 `conv2d` 为例介绍配置文件选项：

- name: *list*
    </br> 函数名字。
    </br> conv2d 在生成基准数据中使用到的函数名字：
    > torch.nn.functional.conv2d(*input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1*) -> Tensor

    </br> 在测试算子适配中使用到的 python 函数名字：
    > diopi_funtions.conv2d(*input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1*) -> Tensor
- atol: *float*
    </br> 用于结果验证的绝对误差参数。
- rtol: *float*
    </br> 用于结果验证的相对误差参数。
    </br> 结果验证使用此函数检查 input 和 other 是否满足条件：
    > |input - other| ≤ atol + rtol × |other|
- dtype: *list*
    </br> 函数张量参数的数据类型。
- tensor_para: *dict*
    </br> 函数张量参数（比如 input，weight，bias 等）。
    </br> args 中：
    + ins: *list*
        </br> 张量参数的名字。
    + shape: *tuple*
        </br> 张量参数的形状。
    + gen_fn（缺省）: *builtin_function*
        </br> 数据生成器，使用了 numpy.random.randn 作为默认生成器。
    
    </br> args 中包含`多组`测试用例张量，shape 元素个数代表张量组数，每个 ins 的 shape 都是`一一对应`的。
    </br> conv_2d 中描述了三组输入张量测试用例：
    </br> * 第一组：`group0 = {"input": tensor(2, 256, 200, 304), "weight": tensor(12, 256, 1, 1), "bias": tensor(12),}`
    </br> * 第二组：`group1 = {"input": tensor(2, 2048, 64, 64), "weight": tensor(2048, 1, 3, 3), "bias": None,}`
    </br> * 第三组：`group2 = {"input": tensor(2, 2048, 1, 1), "weight": tensor(512, 2048, 1, 1), "bias": None,}`
- para: *dict*
    </br> 函数参数（非张量参数）。
    </br> para 中的参数与 tensor_para 中 shape `一一对应`，代表每一组输入张量的函数参数。
    </br> conv_2d 中为每组张量用例描述了函数参数：
    </br> * 第一组：`group0.update(dict(stride=2, padding=0, dilation=1, groups=1))`
    </br> * 第二组：`group1.update(dict(stride=1, padding=12, dilation=12, groups=2048))`
    </br> * 第三组：`group2.update(dict(stride=1, padding=0, dilation=1, groups=1))`


```python
'conv_2d': dict(
    name=["conv2d"],
    atol=1e-5,
    rtol=1e-4,
    dtype=[Dtype.float32, Dtype.float16],
    tensor_para=dict(
        args=[
            {
                "ins": ["input"],
                "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1)),
            },
            {
                "ins": ["weight"],
                "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1)),
            },
            {
                "ins": ["bias"],
                "shape": ((12, ), None, None),
            },
        ]
    ),
    para=dict(
        stride=[2, 1, 1],
        padding=[0, 12, 0],
        dilation=[1, 12, 1],
        groups=[1, 2048, 1],
    ),
),
```

## 适配流程
以 [impl/cuda](impl/cuda) 中 `ReLU` 为例介绍接入一款芯片进行算子开发和测试流程（部分宏定义和函数没有给出实现，用户根据需求自行定义函数）。
### i. 实现运行时函数
测试套件提供了运行时所需 [C-API 函数声明](include/diopi_register.h)，用户根据函数声明实现运行时所需函数，然后进行注册。以 CUDA 设备间内存拷贝为例：
- CUDA 设备间内存拷贝实现
```c
int32_t cuda_memcpy_d2d_async(diopiStreamHandle_t stream_handle,
                              void* dst, const void* src, uint64_t bytes) {
    cudaStream_t phStream = (cudaStream_t)stream_handle;
    CALL_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, phStream));
    return diopiSuccess;
}
```

- 实现函数后进行注册
```c++
int32_t initLibrary() {
    // others register function...
    diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
    // others register function...

    return diopiSuccess;
}
```
### ii. 实现 DIOPI 函数接口
[DIOPI](https://github.com/ParrotsDL/DIOPI/blob/master/include/diopi/functions.h) 提供了了 `ReLU` 函数接口
> DIOPI_API diopiError_t diopiRelu(*diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input*);

使用 cuDNN 函数库实现声明的 `ReLU` 函数接口，具体实现参考[这里](impl/cuda/functions.cpp)，伪代码如下：
```c
extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx,
    diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    // others ...

    DIOPI_CALLCUDNN(cudnnActivationForward(
        handle, descAct, &alpha, desc,
        trIn.data(), &beta, desc, trOut.data()));

    // others ...
    return diopiSuccess;
}
```
### iii. 对适配算子正确性验证
算子完成后需要进行测试验证，在[开始](#start)已经介绍了如何进行算子测试，这里不再进行介绍。
