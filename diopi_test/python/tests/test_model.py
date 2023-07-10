import numpy as np
import os
import conformance as cf
import pytest
from conformance.utils import error_counter, DiopiException, logger, write_report


@pytest.mark.skip("precision rules are not ready")
class TestModel(object):

    def _test_model(self, model_list, test_all):
        logger.info(test_all)
        if test_all:
            test_case = model_list
        else:
            length = len(model_list)
            test_case = [model_list[np.random.randint(low=0, high=length)]]
        for ele in test_case:
            logger.info(f"{ele} model test starts...")
            cf.GenInputData.run("all_ops", ele, [])
            cf.GenOutputData.run("all_ops", ele, [])
            cf.ConformanceTest.run("all_ops", ele, [])
            os.system(f"du -h data/ && rm -rf data/{ele}")
            logger.info(f"Error : {error_counter}")
            write_report()
            if error_counter[0] != 0:
                raise DiopiException(str(error_counter[0]) + " errors during this program")

    def test_cv_models(self, test_all):
        # densenet is too large to use during ci test
        cv_model_list = ['mobilenet_v2', 'resnet50', 'vgg16', 'resnet101', 'seresnet50',
                         'efficientnet', 'shufflenet_v2', 'repvgg', 'swin_transformer', 'vit', 'inceptionv3']
        self._test_model(cv_model_list, test_all)

    def test_det_models(self, test_all):
        det_model_list = ['retinanet', 'faster_rcnn_r50', 'ssd300', 'yolov3', 'atss', 'fcos', 'mask_rcnn',
                          'solo', 'centernet', 'cascade_rcnn', 'detr']
        self._test_model(det_model_list, test_all)

    def test_seg_models(self, test_all):
        seg_model_list = ['unet', 'upernet', 'pspnet', 'fcn', 'deeplabv3', 'deeplabv3plus']
        self._test_model(seg_model_list, test_all)

    def test_other_models(self, test_all):
        # sar is too large to use during ci test
        other_model_list = ['dbnet', 'stgcn', 'crnn', 'hrnet', 'deeppose', 'tsn', 'slowfast']
        self._test_model(other_model_list, test_all)
