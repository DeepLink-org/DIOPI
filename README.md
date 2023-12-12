<div align=center>
<img src="img/deepLink_logo.png">
</div>

# 介绍

DIOPI-设备无关算子接口（Device-Independent Operator Interface, DIOPI）在框架和芯片计算库之间定义了统一的**标准接口**。
旨在训练框架和人工智能芯片之间定义了一套计算契约，良好的函数抽象使得上（框架）下（芯片）两层在适配工程实施时能有效地解耦。
基于这套契约训练框架和人工智能芯片可以独立开发，并将下层芯片适配的工作复用到不同的训练框架适配中去，可降低芯片+框架的适配成本，保障算子实现正确性。

其主要的核心功能如下：
1. **提供300+个标准算子接口，包含LLaMa大模型算子接口**。涵盖了大模型、分类、检测、分割及姿态估计等多个领域深度学习模型所需训练算子。
2. **提供统一的标准算子接口，接入7款硬件芯片**。是训练框架和硬件芯片的“桥梁”，降低训练框架和硬件芯片之间的适配成本，创造更好的国产训练生态。
3. **提供标准测试套件，支持11000+个常见算子测例**，为硬件芯片实现的算子库提供调试验证功能。


## 结构说明

![结构](https://deeplink.readthedocs.io/zh_CN/latest/_images/DIOPI_structure.png)

DIOPI主要包含以下几个组件：

- [proto](https://github.com/DeepLink-org/DIOPI/tree/main/proto)：声明了一套运行时函数接口(diopirt)和标准算子接口(function)。
- [impl](https://github.com/DeepLink-org/DIOPI/tree/main/impl)：对接硬件芯片。硬件厂商可在其中使用硬件软件栈提供的计算接口，实现算子功能。其使用 ```proto/include/diopi/diopirt.h``` 提供的接口实现 ```proto/include/diopi/functions.h``` 声明的标准算子, 并编译为 ```libdiopi_impl.so``` 动态库。
- [diopi_test](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test)：用于保证算子功能正确性。实现 ```proto/include/diopi/diopirt.h``` 声明基础运行时函数，并调用 ```libdiopi_impl.so``` 进行测试验证。
- [adaptor](https://github.com/DeepLink-org/DIOPI/tree/main/adaptor)：用于提供辅助功能函数。目前提供的功能包括自动类型转换、内存分布转换等。



# Quick Start

## 仓库下载
如需在硬件芯片中进行计算接口算子实现，可进行以下步骤（具体参考 [README](https://github.com/DeepLink-org/DIOPI#readme)）。


1. 需下载 [DIOPI仓库](https://github.com/DeepLink-org/DIOPI)，可使用命令：
    ```
    git clone https://github.com/DeepLink-org/DIOPI.git
    ```

    如遇到权限问题，可以参考[FAQ-权限问题](https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/FAQ.html)


## 算子编译


1. 在设备相关目录下提供相应的编译文件，通过脚本进行编译, 以cuda为例：
    ```
    cd impl && sh scripts/build_impl.sh torch
    ```
    或者参考以下命令示例编译 impl：
    ```
    cd impl && mkdir build && cd build && cmake .. -DIMPL_OPT=torch && make -j32
    ```
## 更新基准数据

1. 进入python目录，生成基准数据(需准备 nv 机器和 pytorch2.0 环境)
    ```
    cd python && python main.py --mode gen_data
    ```
    如需指定模型：
    ```
    python main.py --mode gen_data --model_name xxx
    ```
    其中支持的模型名和对应的算子可以通过如下命令获得：
    ```
    python main.py --get_model_list
    ```
    如果想只生成某一个算子的测例可以使用如下命令, 以add系列的算子为例：
    ```
    python main.py --mode gen_data --fname add
    ```


## 校验算子
1. 将数据拷贝到芯片机器上，执行以下命令验证算子：
    ```
    python main.py --mode gen_case  # 生成pytest测例
    python main.py --mode run_test  # 执行测试
    ```
    如需指定模型，以resnet50为例：
    ```
    python main.py --mode gen_case --model_name resnet50 --case_output_dir gencases/resnet50_case
    python main.py --mode run_test --test_cases_path gencases/resnet50_case
    ```
    如需指定某个算子， 以add为例：
    ```
    python main.py --mode gen_case --fname add
    python main.py --mode run_test --test_cases_path gencases/diopi_case
    ```
    如需过滤不支持的数据类型以及部分测试使用nhwc格式张量(如跳过float64以及int64测例)：
    ```
    python main.py --mode gen_case --nhwc
    python main.py --mode run_test --filter_dtype float64 int64
    ```
    可以查看[diopi_test Readme](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test#readme) 了解更详细的设置


2. 验证结果分析
    
    ```
    collecting ... collected 1 items

    gencases/diopi_case/test_diopi_add_add.py::TestMdiopiSaddFadd::test_add_0 PASSED [100%]
    ```
    如需输出HTML格式报告：
    ```
    pip install pytest-testreport
    python main.py --mode run_test --html_report
    ```

## Learn More
组件介绍
* [proto Readme](https://github.com/DeepLink-org/DIOPI/tree/main/proto#readme)
* [impl Readme](https://github.com/DeepLink-org/DIOPI/tree/main/impl#readme)
* [diopi_test Readme](https://github.com/DeepLink-org/DIOPI/tree/main/diopi_test#readme)
* [adaptor Readme](https://github.com/DeepLink-org/DIOPI/tree/main/adaptor#readme)
<!--* [DIPU-Adapter Readme](DIPU-Adapter.md)-->

其他文档
* [DIOPI教程](https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/Introduction.html)
* [C API文档](https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/API/API_index.html)
* [一致性测试 API](https://deeplink.readthedocs.io/zh_CN/latest/DIOPI/diopi_test/python/docs/source/cn_ref.html)
* [常见问题](https://deeplink.readthedocs.io/zh_CN/latest/doc/DIOPI/FAQ.html)
* [Release Note](https://github.com/DeepLink-org/DIOPI/releases)
* [开发者指南](https://github.com/DeepLink-org/DIOPI/blob/main/Contributors.md)
