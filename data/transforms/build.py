# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.RandomCrop(size=cfg.INPUT.SIZE_TRAIN, padding=cfg.INPUT.SIZE_PADDING),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
