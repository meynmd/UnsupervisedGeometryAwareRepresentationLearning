import os
import random
from glob import glob
import random

import numpy as np
import imageio
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision


class Image256toTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(np.array(pic).transpose((2, 0, 1))).float()
        img = img.div(255)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SquarePad(object):
    def __call__(self, image):
        w, h = image.size
        larger = max(w, h)
        square = Image.new(image.mode, (larger, larger), 0)
        square.paste(image, ((larger - w)//2, (larger - h)//2))
        return square


class Real28Dataset(Dataset):
    def __init__(self, real28_root, mean=(0.485, 0.456, 0.406),
                 stdDev=(0.229, 0.224, 0.225), shuffle=True):

        self.gallery_dir = os.path.join(real28_root, 'gallery')
        self.query_dir = os.path.join(real28_root, 'query')
        self.data, subject_ids = self.enumerate_data()
        self.subject_ids = list(subject_ids)
        self.id2index = {sid: i for i, sid in enumerate(self.subject_ids)}
        self.n_images = len(self.data)
        self.mean = mean
        self.std_dev = stdDev
        self.transform_in = torchvision.transforms.Compose([
            SquarePad(),
            torchvision.transforms.Resize((128, 128)),
            Image256toTensor(),
            torchvision.transforms.Normalize(self.mean, self.std_dev)
        ])

    def enumerate_data(self, shuffle=True):

        def find_data(img_dir):
            data = []
            subj_ids = set()
            for img_path in glob(os.path.join(img_dir, '*.jpg')):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                descriptor = img_name.split('_')
                assert len(descriptor) == 4, 'unexpected image name format'
                sid, camera, clothes, _ = [int(v) for v in descriptor]
                subj_ids.add(sid)
                data.append((img_path, sid))
            return data, subj_ids

        query_data, query_ids = find_data(self.query_dir)
        gallery_data, gallery_ids = find_data(self.gallery_dir)

        data = query_data + gallery_data
        if shuffle:
            random.shuffle(data)

        return data, query_ids | gallery_ids

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img_path, subj_id = self.data[idx]
        img = np.array(self.transform_in(Image.open(img_path)),
                       dtype='float32')
        return img, subj_id
