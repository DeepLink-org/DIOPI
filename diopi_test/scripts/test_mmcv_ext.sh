# !/bin/bash
set -e

if [ $# -ne 1 ]
then
    echo Usage: test_mmcv_ext.sh DEVICE
    exit 1
fi

DEVICE=${1}

if [[ $DEVICE == "CUDA" ]]; then
    MMCV_TEST_LIST=(test_active_rotated_filter.py \
    test_assign_score_withk.py \
    test_bbox.py \
    test_deform_roi_pool.py \
    test_knn.py \
    test_convex_iou.py \
    test_min_area_polygons.py \
    test_prroi_pool.py \
    test_chamfer_distance.py \
    test_border_align.py
    )
elif [[ $DEVICE == "MLU" ]]; then
    MMCV_TEST_LIST=()
else
    echo DEVICE $DEVICE not supported!
    exit 1
fi

cd third_party/mmcv_diopi
export PYTHONPATH=${PWD}:$PYTHONPATH
cd tests/test_ops

for elem in ${MMCV_TEST_LIST[@]}
do
    python -m pytest $elem
done
