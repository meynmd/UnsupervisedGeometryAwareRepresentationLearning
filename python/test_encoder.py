import os
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from datasets.prcc_dataset import PRCCDataset
# from build import get_model
from nvs_utils import io as utils_io
import models.unet_encode3D as unet_encode3D


def build_network(encoder_type):
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
    config_dict['encoderType'] = encoder_type

    output_types = config_dict['output_types']

    use_billinear_upsampling = config_dict.get('upsampling_bilinear', False)
    lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
    upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'

    from_latent_hidden_layers = config_dict.get('from_latent_hidden_layers', 0)
    num_encoding_layers = config_dict.get('num_encoding_layers', 4)

    num_cameras = 4
    if config_dict['active_cameras']:  # for H36M it is set to False
        num_cameras = len(config_dict['active_cameras'])

    if lower_billinear:
        use_billinear_upsampling = False
    network_single = unet_encode3D.unet(dimension_bg=config_dict['latent_bg'],
                                        dimension_fg=config_dict['latent_fg'],
                                        dimension_3d=config_dict['latent_3d'],
                                        feature_scale=config_dict['feature_scale'],
                                        shuffle_fg=config_dict['shuffle_fg'],
                                        shuffle_3d=config_dict['shuffle_3d'],
                                        latent_dropout=config_dict['latent_dropout'],
                                        in_resolution=config_dict['inputDimension'],
                                        encoderType=config_dict['encoderType'],
                                        is_deconv=not use_billinear_upsampling,
                                        upper_billinear=upper_billinear,
                                        lower_billinear=lower_billinear,
                                        from_latent_hidden_layers=from_latent_hidden_layers,
                                        n_hidden_to3Dpose=config_dict['n_hidden_to3Dpose'],
                                        num_encoding_layers=num_encoding_layers,
                                        output_types=output_types,
                                        subbatch_size=config_dict['useCamBatches'],
                                        implicit_rotation=config_dict['implicit_rotation'],
                                        skip_background=config_dict['skip_background'],
                                        num_cameras=num_cameras,
                                        )
    return network_single


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

    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    args = ap.parse_args()

    model = build_network(encoder_type)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()

    datasets, dataloaders = get_dataloaders(data_path, batch_size=batch_size)

    accuracy_calculator = AccuracyCalculator(k=1)

    accuracies = test(model, datasets, accuracy_calculator)

    print("Test set accuracy (Precision@1) = "
          "{}".format(accuracies["precision_at_1"]))
    for k, v in accuracies.items():
        print('\t{}: {}'.format(k, v))


if __name__ == "__main__":
    main()


