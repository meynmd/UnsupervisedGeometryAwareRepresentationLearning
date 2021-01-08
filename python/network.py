import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import numpy as np
import numpy.linalg as la
import IPython

from nvs_utils import io as utils_io
from nvs_utils import datasets as utils_data
from nvs_utils import plotting as utils_plt
from nvs_utils import skeleton as utils_skel
import nvs_utils.training as utils_train

import models.unet_encode3D as unet_encode3D


def load_network(config_dict):
    output_types = config_dict['output_types']
    device = 'cuda:0'

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

    if 'pretrained_network_path' in config_dict.keys():  # automatic
        if config_dict['pretrained_network_path'] == 'MPII2Dpose':
            pretrained_network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/output_save/CVPR18_H36M/TransferLearning2DNetwork/h36m_23d_crop_relative_s1_s5_aug_from2D_2017-08-22_15-52_3d_resnet/models/network_000000.pth'
            print("Loading weights from MPII2Dpose")
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0,
                                                 add_prefix='encoder.')  # last argument is to remove "network.single" prefix in saved network
        else:
            print("Loading weights from config_dict['pretrained_network_path']")
            pretrained_network_path = config_dict['pretrained_network_path']
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single,
                                                 submodule=0)  # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_network_path']")

    if 'pretrained_posenet_network_path' in config_dict.keys():  # automatic
        print("Loading weights from config_dict['pretrained_posenet_network_path']")
        pretrained_network_path = config_dict['pretrained_posenet_network_path']
        pretrained_states = torch.load(pretrained_network_path, map_location=device)
        utils_train.transfer_partial_weights(pretrained_states, network_single.to_pose,
                                             submodule=0)  # last argument is to remove "network.single" prefix in saved network
        print("Done loading weights from config_dict['pretrained_posenet_network_path']")
    return network_single

def setup_network():
    config_dict_module = utils_io.loadModule("configs/config_test_encodeDecode.py")
    config_dict = config_dict_module.config_dict
    config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)
    return load_network(config_dict)
