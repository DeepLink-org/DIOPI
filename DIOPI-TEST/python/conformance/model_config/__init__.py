# 12 models for classfication (maybe rm r101)
from .cv_configs.resnet50_config import resnet50_config  # 5.2G
from .cv_configs.resnet101_config import resnet101_config  # 5.0G
from .cv_configs.mobilenet_v2_config import mobilenet_v2_config  # 4.3G
from .cv_configs.efficientnet_config import efficientnet_config  # 12G
from .cv_configs.seresnet50_config import seresnet50_config  # 5.7G
from .cv_configs.densenet_config import densenet_config  # large data
from .cv_configs.vgg16_config import vgg16_config  # 9.8G
from .cv_configs.repvgg_config import repvgg_config  # 4.1G
from .cv_configs.shufflenet_v2_config import shufflenet_v2_config  # 1.9G
from .cv_configs.swin_transformer_config import swin_transformer_config  # 21G
from .cv_configs.vit_config import vit_config  # 4.1G
from .cv_configs.inceptionv3_config import inceptionv3_config  # 8G

# 6 models for segmentation
from .seg_configs.unet_config import unet_config  # 32G
from .seg_configs.upernet_config import upernet_config  # 11G
from .seg_configs.fcn_config import fcn_config  # 5.5G
from .seg_configs.pspnet_config import pspnet_config  # 9.7G
from .seg_configs.deeplabv3_config import deeplabv3_config  # 9.5G
from .seg_configs.deeplabv3plus_config import deeplabv3plus_config  # 14G

# 11 models for detetcion (miss repeat)
from .det_configs.faster_rcnn_r50_config import faster_rcnn_r50_config  # 16G
from .det_configs.retinanet_config import retinanet_config  # 14G
from .det_configs.ssd300_config import ssd300_config  # 4.6G
from .det_configs.yolov3_config import yolov3_config  # 3.9G
from .det_configs.atss_config import atss_config  # 13G
from .det_configs.fcos_config import fcos_config  # 11G
from .det_configs.cascade_rcnn_config import cascade_rcnn_config  # 16G
from .det_configs.mask_rcnn_config import mask_rcnn_config  # 16G
from .det_configs.detr_config import detr_config  # 12G
from .det_configs.centernet_config import centernet_config  # 7.7G
from .det_configs.solo_config import solo_config  # 24G

# 8 models for action/pose
from .other_configs.deeppose_config import deeppose_config  # 10G
from .other_configs.hrnet_config import hrnet_config  # 7G
from .other_configs.stgcn_config import stgcn_config  # 7.1G
from .other_configs.sar_config import sar_config  # large shape
from .other_configs.dbnet_config import dbnet_config  # 14G
from .other_configs.crnn_config import crnn_config  # 2.1G
from .other_configs.slowfast_config import slowfast_config  # 18G
from .other_configs.tsn_config import tsn_config  # 20G


__all__ = ['resnet50_config',
           'resnet101_config',
           'mobilenet_v2_config',
           'efficientnet_config',
           'seresnet50_config',
           'densenet_config',
           'vgg16_config',
           'repvgg_config',
           'shufflenet_v2_config',
           'swin_transformer_config',
           'vit_config',
           'inceptionv3_config',
           # seg
           'unet_config',
           'upernet_config',
           'fcn_config',
           'pspnet_config',
           'deeplabv3_config',
           'deeplabv3plus_config',
           # det
           'faster_rcnn_r50_config',
           'retinanet_config',
           'ssd300_config',
           'yolov3_config',
           'atss_config',
           'fcos_config',
           'cascade_rcnn_config',
           'solo_config',
           'mask_rcnn_config',
           'detr_config',
           'centernet_config',
           # other
           'deeppose_config',
           'hrnet_config',
           'stgcn_config',
           'sar_config',
           'dbnet_config',
           'crnn_config',
           'slowfast_config',
           'tsn_config']
