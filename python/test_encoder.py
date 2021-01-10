import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from datasets.prcc_dataset import PRCCDataset
from build import get_model


def get_dataloaders(data_path, batch_size=32, num_workers=8,
                    mean=(0.485, 0.456, 0.406),
                    std_dev= (0.229, 0.224, 0.225)):

    dataloaders = {}
    datasets = {}
    for camera in ('A', 'C'):
        dataset = PRCCDataset(data_path, 'test', mean, std_dev, camera=camera)
        datasets[camera] = dataset
        dataloaders[camera] = DataLoader(dataset, batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

    return datasets, dataloaders


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(model, datasets, accuracy_calculator):
    query_embeddings, query_labels = get_all_embeddings(datasets['C'], model)
    ref_embeddings, ref_labels = get_all_embeddings(datasets['A'], model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(query_embeddings,
                                                  ref_embeddings,
                                                  np.squeeze(query_labels),
                                                  np.squeeze(ref_labels),
                                                  False)
    return accuracies


def main(encoder_type='UNet', device=torch.device("cuda:0"), batch_size=128,
         data_path='/proj/llfr/staff/mmeyn/briar/data/prcc'):

    datasets, dataloaders = get_dataloaders(data_path, batch_size=batch_size)
    model = get_model(encoder_type)
    accuracy_calculator = AccuracyCalculator(k=1)

    accuracies = test(model, datasets['A'], datasets['C'],
                      accuracy_calculator)

    print("Test set accuracy (Precision@1) = "
          "{}".format(accuracies["precision_at_1"]))
    for k, v in accuracies.items():
        print('\t{}: {}'.format(k, v))


if __name__ == "__main__":
    main()


