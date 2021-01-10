import os
import random
from glob import glob

import numpy as np
import imageio
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision


CAMERAS = {'A', 'B', 'C'}
SPLITS = {'train', 'val', 'test'}

SUBJ_IDS = [
    1, 2, 4, 5, 6, 7, 8, 18, 28, 30, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 75, 91, 92, 
    94, 95, 96, 97, 98, 99, 102, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119, 120, 121, 
    122, 123, 124, 125, 126, 127, 128, 143, 146, 147, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 162,
    163, 164, 165, 167, 168, 169, 170, 171, 172, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 
    189, 190, 193, 194, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 237, 240, 242,
    243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 
    265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 309, 310, 311, 312, 313, 314, 315, 
    316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332
]


class Image256toTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(np.array(pic).transpose((2, 0, 1))).float()
        img = img.div(255)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


#class SquarePad(object):
#    def __call__(self, image_tensor):
#        h, w = img_size = image_tensor.shape[1:]
#        larger_dim = np.argmax(img_size)
#        larger_size = img_size[larger_dim]
#        pad_l, pad_r = (larger_size - w)//2, int(np.ceil((larger_size - w)/2))
#        pad_t, pad_b = (larger_size - h)//2, int(np.ceil((larger_size - h)/2))
#        pad_amt = (pad_l, pad_r, pad_t, pad_b)
#        if larger_dim == 1:
#            tb = torch.cat((image_tensor[:, 0, :], image_tensor[:, -1, :]), dim=1)
#            pad_val = torch.mean(tb).item()
#        else:
#            lr = torch.cat((image_tensor[:, :, 0], image_tensor[:, :, -1]), dim=1)
#            pad_val = torch.mean(lr).item()
#
#        return F.pad(image_tensor, pad_amt, mode='constant', value=pad_val)

class SquarePad(object):
    def __call__(self, image):
        w, h = image.size
        larger = max(w, h)
        square = Image.new(image.mode, (larger, larger), 0)
        square.paste(image, ((larger - w)//2, (larger - h)//2))
        return square


class PRCCDataset(Dataset):
    def __init__(self, prcc_root, split, mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225)):

        assert split in SPLITS
        self.split = split
        self.data_dir = os.path.join(prcc_root, 'rgb', split)
        self.img_dirs = glob(os.path.join(self.data_dir, '*'))
        self.subj_ids = SUBJ_IDS
        self.subj_index = {sid: idx for idx, sid in enumerate(self.subj_ids)}
        self.data = self.enumerate_data()
        self.n_images = len(self.data)
        self.mean = mean
        self.std_dev = stdDev
        #self.transform_in = torchvision.transforms.Compose([
        #    Image256toTensor(),
        #    torchvision.transforms.Normalize(self.mean, self.std_dev),
        #    SquarePad(),
        #    torchvision.transforms.Resize((128, 128))
        #])
        self.transform_in = torchvision.transforms.Compose([
            SquarePad(),
            torchvision.transforms.Resize((128, 128)),
            Image256toTensor(),
            torchvision.transforms.Normalize(self.mean, self.std_dev)
        ])

    def enumerate_data(self):
        data = []
        for subj_id in self.subj_ids:
            images = glob(os.path.join(self.data_dir, str(subj_id), '*.jpg'))
            data += [(img, self.subj_index[subj_id]) for img in images]
        random.shuffle(data)
        return data

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        img_path, subj_id = self.data[idx]
        img = np.array(self.transform_in(Image.open(img_path)),
                       dtype='float32')
        return img, subj_id
