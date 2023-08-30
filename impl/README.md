<div align=center>
<img src="../img/deepLink_logo.png">
</div>


# IMPL

 IMPL 主要用于芯片厂商基于 PROTO 进行标准算子实现，芯片厂商可通过封装自身计算库或者调用 ``kernel`` 的方式来实现 PROTO 定义良好的标准算子接口以备后续测试调用和训练框架调用。

 其价值体现在以实现统一接口计算库的形式，来对接不同训练框架。无需考虑不同训练框架特性，可更专注于提升每个功能性算子的性能。

其主要功能如下：
 * 实现 PROTO 函数接口并编译生成计算库以供测试和训练框架调用


## **实现原理**

#### 实现 TEST 所需运行时函数

  ```diopi_test/include/diopi_register.h``` 中提供了运行时所需 C-API 函数声明，用户根据函数声明实现运行时所需函数，以便测试套件能够在芯片上管理内存等资源。该实现部分仅供测试时使用。

<!-- #### 要求实现并注册的函数列表如下

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
#### 实现函数后进行注册

  实现上述 TEST 所需运行时函数后，通过 `diopi_test/csrc/litert.cpp` 提供的注册函数在 `initLibrary` 中进行注册。示例如下:

  ```
  int32_t initLibrary() {
      // others register function...
      diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
      // others register function...
      return diopiSuccess;
  }
  ``` -->

#### 实现 DIOPI 函数接口

  `proto/include/diopi/functions.h` 根据模型训练和框架开发经验定义了一套标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。截止目前，从30个常用模型所需算子的角度出发，定义了所需的常见训练算子。该实现部分会由 TEST 测试后接入训练框架，用于真实模型训练。在实现的过程中，芯片厂商可根据自身特性来优化算子的性能。

  另外，PROTO 提供了如张量，标量等基础数据结构，这些基础数据结构也出现在DIOPI标准算子的参数列表中。而其中一些数据接口如张量 *Tensor*，上下文 *Context* 是不透明数据类型 ***Opaque data type***。 因此 `proto/include/diopi/diopirt.h` 提供了一套接口用以获取 *Tensor* 的相关信息或者从上下文 *Context* 请求资源。这套接口设计旨在连接训练框架和 DIOPI 算子库， 由训练框架提供给 DIOPI 算子库。而 TEST 将以仅为测试服务的原则实现这套接口。

#### 配置 DIOPI 转换逻辑（可选）


  如果某些算子支持的类型或者layout有限制，可以通过编写配置文件实现调用接口前后的自动转换，转换依赖两个DIOPI接口：`diopiDtypeCast`和`diopiCopyInp`，因此必须实现这两个接口。需要注意的是，由于这种转换是通过copy来完成的，所以会有一定的性能损耗。

  在impl/设备文件夹下新建`convert_config.yaml`文件，配置内容参考：
  
  ```
  - common_config:
    dtype: (int64)->int32, (float64)->float32
    layout: NHWC
    
  - diopiAdd:
      dtype: (int64)->int32, (float64)->float32
      tensor_dtype: 
          input：(float64)->float32
          other：(float64，int64)->float32
          out: (float64，int64)->float32
      layout: NCHW，NHWC, input(NHWC)
  ```

  配置应用可分为三级：
  每个设备通用的配置(`common_config`)，说明了不支持的dtype以及转换规则、接收的layout(默认为NHWC、NCHW)，该配置作用于所有算子。
  每个算子可以有自己的配置(`diopiAdd`)，所有输入输出参数，其中缺省的部分沿用通用配置。
  每个参数也可以有自己的配置(`tensor_dtype`)：对于特殊的参数可以配置参数粒度的，此时会覆盖算子粒度的配置。

  ##### **配置项说明**

  1. **dtype**

  可在设备通用配置和算子配置中设置支持的`dtype`，通用配置的选项包括：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bool，算子内各参数可配置的类型为该参数支持的所有类型。
  ```  
  dtype: (int64)->int32, (float64)->float32
  ```
  括号中为不支持的类型，`->`指向转换后的类型，括号中可以有多个类型，表示这些类型都会转换至`->`后的类型。转换规则可不配置，默认 `dtype: int64->int32` 

  2. **layout**

  layout可配置的选项包括NHWC和NCHW，后续若有其他layout，DIOPI支持后也可配置。配置中两个可同时包含，表示两种类型都支持，默认值即为都支持，对layout没有特殊要求。layout也可以配置算子和参数粒度的，配置形式如下：
  ```    
  layout: NCHW，NHWC, input(NHWC)
  ```
