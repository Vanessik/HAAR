from functools import partial
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from openaimodel import UNetModel
from src.utils import layers

sys.path.append(os.path.join(os.getcwd(), 'submodules/k-diffusion'))
from k_diffusion import models, utils

import json
import math
from pathlib import Path

from jsonmerge import merge



def round_to_power_of_two(x, tol):
    approxs = []
    for i in range(math.ceil(math.log2(x))):
        mult = 2**i
        approxs.append(round(x / mult) * mult)
    for approx in reversed(approxs):
        error = abs((approx - x) / x)
        if error <= tol:
            return approx
    return approxs[0]


def load_config(path_or_dict):
    defaults_image_v1 = {
        'model': {
            'patch_size': 1,
            'augment_wrapper': True,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
            'cross_cond_dim': 0,
            'cross_attn_depths': None,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-6,
            'weight_decay': 1e-3,
        },
    }
    defaults_image_transformer_v1 = {
        'model': {
            'd_ff': 0,
            'augment_wrapper': False,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-4,
            'betas': [0.9, 0.99],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
    }
    defaults = {
        'model': {
            'sigma_data': 1.,
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'loss_config': 'karras',
            'loss_weighting': 'karras',
            'loss_scales': 1,
        },
        'dataset': {
            'num_classes': 0,

        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
        'lr_sched': {
            'type': 'constant',
            'warmup': 0.,
        },
        'ema_sched': {
            'type': 'inverse',
            'power': 0.6667,
            'max_value': 0.9999
        },
    }
    if not isinstance(path_or_dict, dict):
        file = Path(path_or_dict)
        if file.suffix == '.safetensors':
            metadata = utils.get_safetensors_metadata(file)
            config = json.loads(metadata['config'])
        else:
            config = json.loads(file.read_text())
    else:
        config = path_or_dict
    if config['model']['type'] == 'image_v1':
        config = merge(defaults_image_v1, config)
    elif config['model']['type'] == 'image_transformer_v1':
        config = merge(defaults_image_transformer_v1, config)
        if not config['model']['d_ff']:
            config['model']['d_ff'] = round_to_power_of_two(config['model']['width'] * 8 / 3, tol=0.05)
    return merge(defaults, config)


def make_model(config):

    dataset_cfg = config['dataset']
    config = config['model']
    
    num_classes = dataset_cfg['num_classes']

    if config['type'] == 'image_v1':
        model = models.ImageDenoiserModelV1(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
        )
        if config['augment_wrapper']:
            model = augmentation.KarrasAugmentWrapper(model)
    elif config['type'] == 'image_transformer_v1':
        model = models.ImageTransformerDenoiserModelV1(
            n_layers=config['depth'],
            d_model=config['width'],
            d_ff=config['d_ff'],
            in_features=config['input_channels'],
            out_features=config['input_channels'],
            patch_size=config['patch_size'],
            num_classes=num_classes + 1 if num_classes else 0,
            dropout=config['dropout_rate'],
            sigma_data=config['sigma_data'],
        )

    elif config['type'] == 'openai':
        model = UNetModel(
                            image_size=config['input_size'][0],
                            in_channels=config['input_channels'],
                            model_channels=config['model_channels'],
                            out_channels=config['input_channels'],
                            num_res_blocks=config['num_res_blocks'],
                            attention_resolutions=config['attention_resolutions'],
                            dropout=0,
                            channel_mult=config['channel_mult'],
                            conv_resample=True,
                            dims=2,
                            num_classes=None,
                            use_checkpoint=False,
                            use_fp16=False,
                            num_heads=config['num_heads'],
                            num_head_channels=config['num_head_channels'],
                            num_heads_upsample=-1,
                            use_scale_shift_norm=config['use_scale_shift_norm'],
                            resblock_updown=config['resblock_updown'],
                            use_new_attention_order=False,
                            use_spatial_transformer=config['use_spatial_transformer'],
                            transformer_depth=config['transformer_depth'],
                            context_dim=config['context_dim'],
                            n_embed=None,
                            legacy=config['legacy']
                            )
    else:
        raise ValueError(f'unsupported model type {config["type"]}')

    return model


def make_denoiser_wrapper(config):
    config = config['model']
    sigma_data = config.get('sigma_data', 1.)
    has_variance = config.get('has_variance', False)
    loss_config = config.get('loss_config', 'karras')
    if loss_config == 'karras':
        weighting = config.get('loss_weighting', 'karras')
        scales = config.get('loss_scales', 1)
        if not has_variance:
            return partial(layers.Denoiser, sigma_data=sigma_data, weighting=weighting, scales=scales)
        return partial(layers.DenoiserWithVariance, sigma_data=sigma_data, weighting=weighting)
    if loss_config == 'simple':
        if has_variance:
            raise ValueError('Simple loss config does not support a variance output')
        return partial(layers.SimpleLossDenoiser, sigma_data=sigma_data)
    raise ValueError('Unknown loss config type')