#!/bin/bash
# echo "First argument: $1"


current_time=$(date "+%Y.%m.%d-%H.%M.%S")


DIOPI_TRACK_ACL=1 DIPU_TRACK_ACL=1 DIPU_TRACK_HCCL=1 ASCEND_COREDUMP_SIGNAL=1 ASCEND_GLOBAL_LOG_LEVEL=0 DIPU_TRACK_ACL=1 DIPU_DEBUG_ALLOCATOR=15 python main.py --mode run_test | tee test_unique_${current_time}.log