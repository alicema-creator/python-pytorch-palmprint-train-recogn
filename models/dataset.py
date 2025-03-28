# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        # Replace the deprecated _is_tensor_image check
        if not torch.is_tensor(tensor):
            raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support grayscale image.')

        # Flatten the tensor to normalize non-zero values
        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # Normalize
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        # Reshape back to original dimensions
        tensor = tensor.view(c, h, w)

        # Repeat channels if needed
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


class MyDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing paths & labels of the input images \n
    transforms: None 
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''

    def __init__(self, txt, transforms=None, train=True, imside=128, outchannels=1):
        self.train = train
        self.imside = imside  # 128, 224
        self.chs = outchannels  # 1, 3
        self.text_path = txt
        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(self.imside),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC, expand=False, center=(0.5 * self.imside, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC, expand=False, center=(0.0, 0.5 * self.imside)),
                        ]),
                    ]),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]

        data = Image.open(img_path).convert('L')
        data = self.transforms(data)

        return data, int(label)

    def __len__(self):
        return len(self.images_path)