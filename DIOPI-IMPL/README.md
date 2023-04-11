<div align=center>
<img src="resources/deepLink_logo.png">
</div>



# DIOPI-IMPL

 DIOPI-IMPL 主要用于芯片厂商基于 DIOPI-PROTO 进行标准算子实现，和基于 DIOPI-TEST 注册基础少量的运行时函数。芯片厂商需要在 DIOPI-IMPL 通过注册的形式，为后续测试提供如内存拷贝、流创建销毁等可管理设备芯片的功能，该实现部分仅供 DIOPI-TEST 测试所用。更为重要的是，芯片厂商可通过封装自身计算库或者调用 ```kernel``` 的方式来实现 DIOPI-PROTO 定义良好的标准算子接口以备后续测试调用和训练框架调用。

 其价值体现在以实现统一接口计算库的形式，来对接不同训练框架。无需考虑不同训练框架特性，可更专注于提升每个功能性算子的性能。

其主要功能如下：
 * 实现并注册 DIOPI-TEST 所需运行时函数以供测试使用
 * 实现 DIOPI-PROTO 函数接口并编译生成计算库以供测试和训练框架调用


## **实现原理**

* 实现 DIOPI-TEST 所需运行时函数:

  ```DIOPI-TEST/include/diopi_register.h``` 中提供了运行时所需 C-API 函数声明，用户根据函数声明实现运行时所需函数，然后进行注册，以便测试套件能够在芯片上管理内存等资源。该实现部分仅供测试时使用。

* 要求实现并注册的函数列表如下：
  ```
  typedef int32_t (*create_stream_func_t)(diopiStreamHandle_t*);
  //其中diopiStreamHandle_t为void*类型别名;
  typedef int32_t (*destroy_stream_func_t)(diopiStreamHandle_t);

  typedef void* (*malloc_func_t)(uint64_t);
  typedef void (*free_func_t)(void*);

  typedef int32_t (*memcpy_h2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2h_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);

  typedef int32_t (*sync_stream_func_t)(diopiStreamHandle_t stream);

  typedef const char* (*get_last_error_string_func_t)();
  ```
* 实现函数后进行注册：

  实现上述 DIOPI-TEST 所需运行时函数后，通过 DIOPI-TEST/csrc/litert.cpp 提供的注册函数在 initLibrary 中进行注册。示例如下:

  ```
  int32_t initLibrary() {
      // others register function...
      diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
      // others register function...
      return diopiSuccess;
  }
  ```

* 实现 DIOPI 函数接口:

  DIOPI-PROTO/include/diopi/functions.h 根据模型训练和框架开发经验定义了一套标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。截止目前，从30个常用模型所需算子的角度出发，定义了所需的常见训练算子。该实现部分会由 DIOPI—TEST 测试后接入训练框架，用于真实模型训练。在实现的过程中，芯片厂商可根据自身特性来优化算子的性能。

  另外，DIOPI-PROTO 提供了如张量，标量等基础数据结构，这些基础数据结构也出现在DIOPI标准算子的参数列表中。而其中一些数据接口如张量 *Tensor*，上下文 *Context* 是不透明数据类型 ***Opaque data type***。 因此 DIOPI-PROTO/include/diopi/diopirt.h 提供了一套接口用以获取 *Tensor* 的相关信息或者从上下文 *Context* 请求资源。这套接口设计旨在连接训练框架和 DIOPI 算子库， 由训练框架提供给 DIOPI 算子库。而 DIOPI-TEST 将以仅为测试服务的原则实现这套接口。

## **使用教程**

### 安装

1. 下载DIOPI-Test测验仓库：
    ```
    git clone https://github.com/OpenComputeLab/DIOPI-TEST.git
    ```
2. 使用 DIOPI-IMPL 提供编译文件的编译计算库。以下示例仅供参考：
    ```
    mkdir build && cd build && cmake .. -DIMPL_OPT=cuda && make -j32
    ```
### 算子验证
 1. 确认启动前提

    确保编译生成的计算库 libdiopi_impl.so 以及 DIOPI-TEST 提供的运行时库 libdiopirt.so 保存在 DIOPI-TEST/lib 下。

    确保基准测试数据保存在路径 DIOPI-TEST/python/data下。

2. 算子验证请参考 [DIOPI-Test 使用教学](https://github.com/OpenComputeLab/DIOPI-TEST/blob/main/README.md)
