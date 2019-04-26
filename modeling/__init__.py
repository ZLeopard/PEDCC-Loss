# encoding: utf-8

from .VGG import VGG
from .resnet import resnet_face18
from .metric_loss import  AMSoftmax, CosineLinear_PEDCC


def build_model(cfg):
    if cfg.ARCHI.NAME == 'VGG':
        model = VGG(cfg)
        return model
    elif cfg.ARCHI.NAME == "ResNet":
        model = resnet_face18(True)
        return model
