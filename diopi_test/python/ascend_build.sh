export DIOPI_BUILD_TESTRT=ON
export CPLUS_INCLUDE_PATH=/home/tangding/anaconda3/envs/dipu/include/python3.8:$CPLUS_INCLUDE_PATH
#export CPLUS_INCLUDE_PATH=/usr/local/python3.7.5/include/python3.7m/:$CPLUS_INCLUDE_PATH
cd ../../impl
sh scripts/build_impl.sh clean
sh scripts/build_impl.sh ascend
cd ../diopi_test/python
