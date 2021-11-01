import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
#from utils import is_image_file
import os
from PIL import Image
import random

from datasyn.transforms import to_norm_tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".JPG"])


def default_loader(path):
    return Image.open(path).convert('RGB')

def ToTensor(pic):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backard compability
        return img.float().div(255)
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

# You should build custom dataset as below.
class TEST_DATASET(data.Dataset):
    def __init__(self,dataPath='',loadSize=286,fineSize=256,flip=1):
        super(TEST_DATASET, self).__init__()
        # list all images into a list
        self.listA = [x for x in listdir(dataPath) if is_image_file(x)]
        
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        pathA = os.path.join(self.dataPath,self.listA[index])
        imgA = default_loader(pathA) # 256x256
        imgA = to_norm_tensor(imgA)
        return imgA

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.listA)

import torchdata
BaseDataset = torchdata.Dataset