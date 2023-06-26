# !/bin/bash
set -e

if [ $# -ne 1 ]
then
    echo Usage: test_mmcv_ext.sh DEVICE
    exit 1
fi

DEVICE=${1}
echo $MMCV_TEST_HOME

if [[ $DEVICE == "CUDA" ]]; then
    MMCV_TEST_LIST=("tests/test_ops/test_nms.py -k test_nms_allclose" \
    "tests/test_ops/test_roi_align.py" \
    "tests/test_ops/test_focal_loss.py -k sigmoid" \
    "tests/test_ops/test_voxelization.py" \
    "tests/test_ops/test_modulated_deform_conv.py::TestMdconv::test_mdconv_float" \
    "tests/test_ops/test_modulated_deform_conv.py::TestMdconv::test_mdconv_double"
    )
elif [[ $DEVICE == "MLU" ]]; then
    MMCV_TEST_LIST=()
else
    echo DEVICE $DEVICE not supported!
    exit 1
fi

cd $MMCV_TEST_HOME

for elem in "${MMCV_TEST_LIST[@]}";
do
    python -m pytest $elem
done
