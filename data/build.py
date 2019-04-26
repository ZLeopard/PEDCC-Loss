# encoding: utf-8

from torch.utils import data

from .datasets.cifar100 import CIFAR100
from .datasets.face_dataset import FaceDataset
from .transforms import build_transforms


def build_dataset(cfg, transforms, is_train=True):
    if cfg.DATASETS.NAME == "CIFAR100":
        datasets = CIFAR100(root='./data/CIFAR100', train=is_train, transform=transforms, download=True)
        return datasets
    elif cfg.DATASETS.NAME == "FACE_DATA":
        datasets = FaceDataset(cfg, root='', data_list_file='', phase='train')
        return datasets


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, is_train)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader
