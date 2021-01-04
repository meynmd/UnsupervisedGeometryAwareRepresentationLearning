import os
import random
from glob import glob

import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import torchvision


CAMERAS = {'A', 'B', 'C'}
SPLITS = {'train', 'val', 'test'}


class Image256toTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
        img = img.div(255)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PRCCDataset(Dataset):
    def __init__(self, prcc_root, split, mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225)):

        assert split in SPLITS
        self.split = split
        self.data_dir = os.path.join(prcc_root, 'rgb', split)
        self.img_dirs = glob(os.path.join(self.data_dir, '*'))
        self.subj_ids = map(os.path.basename, self.img_dirs)
        self.data = self.enumerate_data()
        self.n_images = len(self.data)
        self.mean = mean
        self.std_dev = stdDev
        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(),
            torchvision.transforms.Normalize(self.mean, self.std_dev)
        ])

    def enumerate_data(self):
        data = []
        for subj_id in self.subj_ids:
            images = glob(os.path.join(self.data_dir, subj_id, '*.jpg'))
            data += [(img, subj_id) for img in images]
        random.shuffle(data)
        return data

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img_path, subj_id = self.data[idx]
        img = np.array(self.transform_in(imageio.imread(img_path)),
                       dtype='float32')
        return img, subj_id
