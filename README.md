<div align=center>
<img src="resources/deepLink_logo.png">
</div>

## **使用教程(测试)**
1. 编译impl
export DIOPI_BUILD_TESTRT=ON
cd DIOPI-IMPL && sh scripts/build_impl.sh torch
2. 运行算子测试
cd DIOPI-TEST && python python/main.py --mode gen_data --fname relu && python python/main.py --mode run_test --fname relu