import os
from glob import glob

import torch
from torch.utils.data import Dataset

CAMERAS = ['A', 'B', 'C']

SPLITS = {
    'train': ['257', '323', '057', '074', '008', '279', '117', '162', '069',
              '064', '091', '328', '214', '322', '005', '216', '075', '072',
              '188', '264', '096', '097', '118', '242', '263', '272', '212',
              '230', '159', '006', '059', '223', '099', '146', '030', '156',
              '282', '120', '219', '004', '071', '060', '058', '319', '260',
              '309', '326', '073', '001', '321', '007'],
    'val': ['094', '152', '112', '070', '061', '062', '028', '018', '324', '265'],
    'test': ['325', '320', '183', '063', '056', '167', '002', '202', '182', '186']
}

class PRCCDataset(Dataset):
    def __init__(self, prcc_root, split):
        assert split in SPLITS.keys()
        self.split = split
        self.data_dir = os.path.join(prcc_root, 'rgb', 'test')

    def enumerate_data(self):
        data = []
        for subj_id in SPLITS[self.split]:
            for camera in CAMERAS:
                imgs = glob(os.path.join(self.data_dir, camera, subj_id, '*.jpg'))
                data += [(img, subj_id)]


    def __len__(self):
        return self.n_subj*(self.n_subj - 1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()






