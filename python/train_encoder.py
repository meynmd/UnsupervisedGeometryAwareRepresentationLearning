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
    for split in ('train', 'val'):
        dataset = PRCCDataset(data_path, split, mean, std_dev)
        datasets[split] = dataset
        dataloaders[split] = DataLoader(dataset, batch_size,
                                        shuffle=(split == 'train'),
                                        num_workers=num_workers)

    return datasets, dataloaders


def train_epoch(model, train_loader, loss_func, optimizer,
                device, mining_func, epoch=1):

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model.encode_3d(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined "
                  "triplets = {}".format(epoch, batch_idx, loss,
                                         mining_func.num_triplets))


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(model, train_set, test_set, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                train_embeddings,
                                                np.squeeze(test_labels),
                                                np.squeeze(train_labels),
                                                False)
    print("Test set accuracy (Precision@1) = "
          "{}".format(accuracies["precision_at_1"]))
    for k, v in accuracies.items():
        print('\t{}: {}'.format(k, v))
    return accuracies


def main(encoder_type='UNet', max_epochs=50, device=torch.device("cuda:0"),
         batch_size=128, data_path='/proj/llfr/staff/mmeyn/briar/data/prcc'):
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', '-s')
    args = ap.parse_args()

    if args.save is None:
        save_dir = os.path.join(os.path.abspath('.'), 'checkpoints')
    else:
        save_dir = args.save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets, dataloaders = get_dataloaders(data_path, batch_size=batch_size)
    model = get_model(encoder_type)
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance,
                                         reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance,
                                            type_of_triplets="semihard")
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",),
                                             k=1)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        module = name.split(".")[0]
        if module == "encoding_projection":
            classifier_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.Adam([{"params": base_params},
                            {"params": classifier_params, 'lr': 1e-2}], lr=1e-4)

    for epoch in range(1, max_epochs + 1):
        train_epoch(model, dataloaders['train'], loss_func, optimizer,
                    device, mining_func, epoch)

        acc = test(model, datasets['train'], datasets['val'],
                   accuracy_calculator)

        if epoch % 5 == 0:
            checkpoint_name = 'epoch_{:02d}_prec1_{:.4f}.pth'.format(epoch, acc["precision_at_1"])
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
