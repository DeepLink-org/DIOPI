<div align=center>
<img src="resources/deepLink_logo.png">
</div>

## **使用教程**
1. 编译impl
cd DIOPI-IMPL && sh script/build_impl.sh torch
2. 编译test
cd DIOPI-TEST && sh script/build_test.sh torch
3. 运行算子测试
cd DIOPI-TEST && python python/main.py --mode gen_data --fname relu && python python/main.py --mode run_test --fname relu