# 常见问题


#### 1. DIOPI算子开发流程是怎样的？

- 搭建环境：安装芯片厂商SDK和必要的系统工具。
- 添加算子代码：在impl项目相应目录中添加算子c++代码。
- 生成基准数据：执行基准数据生成命令，生成测试时需要的基准数据。
- 算子测试：执行算子测试命令，将自己实现的算子计算结果和基准数据进行对比。

#### 2. 如何搭建DIOPI-impl开发环境？如果在自己的PC中开发，需要安装哪些包，cmakelist中include、lib路径需要修改哪些？

首先机器上要有芯片厂商的软件栈，配好环境变量后CMakelist中的include和lib路径第不用修改的，source完环境后可以直接编译。我们推荐使用conda管理python环境，具体安装的包可以在运行报错时，根据提示安装。

#### 3. 代码的目录结构是怎样的？编译的命令是什么？编译的结果在哪里？

（1）代码目录结构
* DIOPI-Test主要包含impl(算子实现)，diopi运行时文件和一致性测试的代码
* impl中将不同厂商的的算子实现存放在不同的路径下，例如camb对应寒武纪的算子实现

（2）编译指令
    以寒武纪软件栈为例，先source对应环境, 然后使用如下指令进行编译，
    请注意：对应的软件栈不同，则环境和编译选项也有所不同
```
sh scripts/build_impl.sh camb 
```

（3）编译结果位置
```
/DIOPI-Test/build下 
```
    
#### 4. 生成baseline有哪些环境要求？如何生成baseline并进行测试？生成的数据在哪里？如何查看数据的详细内容？

(1) 生成baseline的环境要求

- ```cuda```：需要环境预装好pytorch，安装pytorch可参考[pytorch官网](https://github.com/pytorch/pytorch)

(2) 如何生成baseline并进行测试？

第一步生成基准输入和输出数据，第二步验证适配的算子的正确性。

测试脚本运行命令（在./python目录下）：
```
python main.py [-h] [--mode MODE] [--fname FNAME]
```
选项说明：
- ```--mode``` 可选项：```gen_data```, ```run_test```
运行模式选项，用于选择当前函数生成基准数据还是测试算子
- ```--fname``` 缺省：```all_ops```
函数名字选项，如果指定函数名字（配置文件中测例的 name）则会对该算子进行基准数据生成和测试，不指定默认对所有算子生成基准数据和测试。

例如：
1.  在 Nvidia 设备上生成基准输入和输出数据
```
python main.py --mode gen_data --fname all_ops
```
2. 在接入芯片设备上运行测试
```
python main.py --mode run_test --fname all_ops
```

(3) 生成的数据在哪里？

在```DIOPI-Test/python/data```中，以pickle形式存储

(4)如何查看数据的详细内容？
有两种方式可以查看数据的详细内容
- ```pickle.load()``` 将测试object读取进内存再进行自定义的可视化和操作，pickle相关使用可以参考[页面](https://docs.python.org/3/library/pickle.html)
- 将```DIOPI-Test/python/conformance/utils.py```中```log_level```设置为```DEBUG```
这样在测试中，如果发现异常（如值不对）则会将数据信息打出

#### 5. 如何测试添加的算子是否正确？测试命令是什么？测试结果如何看？如果测试结果不对如何查看更多详细内容？

在README中会有介绍算子测试方法，我们这里使用的是根据```python/conformance/diopi_configs.py```中描述的算子信息在Nvidia机器上生成算子输入以及算子的输出，并将其他芯片厂商的算子运算结果与Nvidia对比。

算子添加后，CI上会进行测试，算子是否正确可看CI日志。测试命令请见README。测试结果会在终端中打印出来。如果结果不正确，可以在```python/conformance/utils.py中将default_cfg_dict[log_level] = DEBUG```。这样会在```python/error_report.csv```中显示详细的错误信息。

---
### 无法找到问题
您可在项目中提交issue，将您遇到的问题告诉我们。
<!-- issue回复的流程可在[开发者指南中](Contributors.md)获取。
2. 或者您也可以加入[开发者社区]()，像我们提供反馈和建议。 -->