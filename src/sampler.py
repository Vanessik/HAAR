import torch
import sys
import os

sys.path.append(os.path.join(os.getcwd(), './submodules/k-diffusion'))
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step, to_d
from tqdm.auto import trange

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, cfg_scale=0, seed=None):
    """Ancestral sampling with Euler method steps."""
    if seed is not None:
        torch.manual_seed(seed) 
        
    extra_args = {} if extra_args is None else extra_args

    cross_cond_zero = extra_args['cross_cond_zero']
    del extra_args['cross_cond_zero']

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        uncond_pred_noise = model(x, sigmas[i] * s_in, cross_cond=cross_cond_zero)

        if cfg_scale > 0:
            denoised = torch.lerp(uncond_pred_noise, denoised, cfg_scale)
        else:
            denoised = uncond_pred_noise

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x
