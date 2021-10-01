import os.path
from os.path import join
from datasyn.image_folder import make_dataset
from datasyn.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from PIL import Image
import random
import torch
import math
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import glob
#import util.util as util
import datasyn.torchdata as torchdata

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(320, 480) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)
    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2


BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    def reset(self):
        if self.shuffle:
            print('Reset Dataset...')
            self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(self, B_path=None,R_path=None, size=None, enable_transforms=True):
        super(CEILDataset, self).__init__()
        self.size = size
        self.enable_transforms = enable_transforms

        # 生成path list
        self.B_paths = glob.glob(B_path)
        self.R_paths = glob.glob(R_path)

        # 限制每个epoch的syn图像多少
        if size is not None:
            self.B_paths = self.B_paths[:size]
            self.R_paths = self.R_paths[:size]

        self.syn_model = ReflectionSythesis_2()
        self.reset(shuffle=True) # 先执行一下

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.B_paths)
            random.shuffle(self.R_paths)
        num_paths = len(self.B_paths) 
        #print('Shuffle Syn!')

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)
        '''
        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)
        '''
        B = to_norm_tensor(t_img)
        R = to_norm_tensor(r_img)
        M = to_norm_tensor(m_img)
        return B, R, M
        
    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)
        
        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
        
        # 消除不合格pair
        flag = 0
        while np.mean(t_img)*1/2 > np.mean(r_img):
            flag+= 1
            print('Recycle the syn images for ilumination')
            index_B = (index+ 1) % len(self.B_paths) 
            index_R = (index+ 1) % len(self.R_paths) 
            B_path = self.B_paths[index_B]
            R_path = self.R_paths[index_R]
            t_img = Image.open(B_path).convert('RGB')
            r_img = Image.open(R_path).convert('RGB')
            if flag==3:
                break

        B, R, M = self.data_synthesis(t_img, r_img)
        return M,B,R

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))

class RIDDataset(BaseDataset):
    def __init__(self, B_path=None,R_path=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        super(RIDDataset, self).__init__()
        self.size = size
        #self.datadir = datadir
        self.enable_transforms = enable_transforms

        # 生成path list
        self.B_paths = glob.glob(B_path)
        self.R_paths = glob.glob(R_path)

        # 限制每个epoch的syn图像多少
        if size is not None:
            self.B_paths = self.B_paths[:size]
            self.R_paths = self.R_paths[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma, low_gamma=low_gamma, high_gamma=high_gamma)
        self.reset(shuffle=True)

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.B_paths)
            random.shuffle(self.R_paths)
        num_paths = len(self.B_paths) 
        #print('Shuffle RID!')

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)
        
        '''
        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)
        '''
        B = to_norm_tensor(t_img)
        R = to_norm_tensor(r_img)
        M = to_norm_tensor(m_img)

        return B, R, M
        
    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)
        
        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')

        B, R, M = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(B_path)
        return M,B,R

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))





####### 后面都没用
class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        
        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = to_tensor(t_img)
        M = to_tensor(m_img)

        dic =  {'input': M, 'target_t': B, 'fn': fn, 'real':True, 'target_r': B} # fake reflection gt 
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))
        
        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1
        
        m_img = Image.open(join(self.datadir, fn)).convert('RGB')

        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class PairedCEILDataset(CEILDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5):
        self.size = size
        self.datadir = datadir

        self.fns = fns or os.listdir(join(datadir, 'reflection_layer'))
        if size is not None:
            self.fns = self.fns[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma)
        self.enable_transforms = enable_transforms
        self.reset()

    def reset(self):
        return

    def __getitem__(self, index):
        fn = self.fns[index]
        B_path = join(self.datadir, 'transmission_layer', fn)
        R_path = join(self.datadir, 'reflection_layer', fn)
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
    
        B, R, M = self.data_synthesis(t_img, r_img)

        data = {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}
        # return M, B
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' %(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size


class RepeatedDataset(BaseDataset):
    def __init__(self, dataset, repeat=1):
        self.dataset = dataset
        self.size = len(dataset) * repeat        
        # self.reset()

    def reset(self):
        
        self.dataset.reset()

    def __getitem__(self, index):
        dataset = self.dataset
        return dataset[index%len(dataset)]
    
    def __len__(self):
        return self.size
