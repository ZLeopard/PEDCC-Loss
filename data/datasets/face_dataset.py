import os
from torchvision import transforms as T
import torchvision
from torch.utils import data
from PIL import Image
import numpy as np

class FaceDataset(data.Dataset):

    def __init__(self, cfg, root, data_list_file, phase='train'):
        self.phase = phase
        self.size = cfg.INPUT.SIZE_TRAIN

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(144),
                T.RandomCrop(self.size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(144),
                T.CenterCrop(self.size),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert('L')     # 进行灰度化
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)
