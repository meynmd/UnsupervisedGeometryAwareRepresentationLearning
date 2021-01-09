from torch.utils.data import DataLoader

from datasets.prcc_dataset import PRCCDataset
from build import get_model


def get_dataloaders(data_path, batch_size=32, num_workers=8,
                    mean=(0.485, 0.456, 0.406),
                    std_dev= (0.229, 0.224, 0.225)):

    dataloaders = {}
    for split in ('train', 'val'):
        dataset = PRCCDataset(data_path, split, mean, std_dev)
        dataloaders[split] = DataLoader(dataset, batch_size,
                                        shuffle=(split == 'train'),
                                        num_workers=num_workers)

    return dataloaders


