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
import cmc_curve as cmc

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
    print("computing train embeddings...")
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    print("computing val embeddings...")
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("computing accuracy...")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                train_embeddings,
                                                np.squeeze(test_labels),
                                                np.squeeze(train_labels),
                                                False)
    print("val set precision@1 = "
          "{}".format(accuracies["precision_at_1"]))

    return accuracies


def main(max_epochs=100, device=torch.device("cuda:0"),
         batch_size=128, data_path='/proj/llfr/staff/mmeyn/briar/data/prcc'):
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', '-e', default='UNet')
    ap.add_argument('--save', '-s')
    ap.add_argument('--resume')
    args = ap.parse_args()

    encoder_type = args.encoder

    if args.save is None:
        save_dir = os.path.join(os.path.abspath('.'), 'checkpoints')
    else:
        save_dir = args.save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datasets, dataloaders = get_dataloaders(data_path, batch_size=batch_size)
    model = get_model(encoder_type)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.1, distance=distance,
                                         reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.1, distance=distance,
                                            type_of_triplets="all")
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",),
                                             k=1)

    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        module = name.split(".")[0]
        if module == "encoding_projection":
            classifier_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.SGD([{"params": base_params},
                           {"params": classifier_params, 'lr': 1e-2}], 
                          lr=1e-5, momentum=0.9)
    
    # dataloaders for testing
    _, test_dl = cmc.get_dataloaders(data_path, batch_size=batch_size)

    for epoch in range(1, max_epochs + 1):
        for i, pg in enumerate(optimizer.param_groups):
            print('parameter group {:02d}, {} parameters, lr: {:.4f}'.format(i, len(pg['params']), pg['lr'])) 
        train_epoch(model, dataloaders['train'], loss_func, optimizer,
                    device, mining_func, epoch)

        lrs = [pg['lr'] for pg in optimizer.param_groups]
        lr_max, lr_min = max(lrs), min(lrs)
        for i, pg in enumerate(optimizer.param_groups): 
            d_lr = lr_max - pg['lr']
            if d_lr > 0:
                pg['lr'] += 0.02 * d_lr
            pg['lr'] *= 0.98

        # acc = test(model, datasets['train'], datasets['val'],
        #            accuracy_calculator)
        print('evaluating...')
        cmc_score = cmc.test(model, test_dl, k=1)
        print('rank 1: {:.4f}'.format(cmc_score[0]))

        if epoch % 5 == 0:
            # checkpoint_name = 'epoch_{:03d}_prec1_{:.4f}.pth'.format(epoch, acc["precision_at_1"])
            checkpoint_name = 'epoch_{:03d}_prec1_{:.4f}.pth'.format(epoch, cmc_score[0])
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
