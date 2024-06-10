from .config import make_model, make_denoiser_wrapper
import torch
import accelerate
from copy import deepcopy
import os
import sys

sys.path.append(os.path.join(sys.path[0], './submodules/k-diffusion'))
import k_diffusion as K


def setup_model(config, ckpt_path, device):
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    inner_model = make_model(config) 
   
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=0)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=1)

    model =  make_denoiser_wrapper(config)(inner_model)
    model_ema = deepcopy(model)
    model_ema.to(device)
    accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
    model_ema.eval()

    return model_ema, accelerator


def setup_train_model(config, ckpt_path, device):
    
    ckpt = torch.load(ckpt_path, map_location='cpu')

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=0)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=1)
    
    unwrap = accelerator.unwrap_model

    inner_model = make_model(config)
    inner_model_ema = deepcopy(inner_model)

    model = make_denoiser_wrapper(config)(inner_model)
    model_ema = make_denoiser_wrapper(config)(inner_model_ema)

    model_ema.to(device)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu')

    unwrap(model.inner_model).load_state_dict(ckpt['model'])
    unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])

    ema_sched_config = config['ema_sched']
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'], max_value=ema_sched_config['max_value'])

    ema_sched.load_state_dict(ckpt['ema_sched'])

    return inner_model, accelerator, model, ema_sched, model_ema