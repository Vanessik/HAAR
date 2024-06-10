import os
import sys
import torch
from tqdm.auto import tqdm
import argparse


sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))
from lavis.models import load_model_and_preprocess

from src.datasets.dataset import Hairstyle
from src.upsampling.upsampler import HairstyleUpsampler

from src.sampler import sample_euler_ancestral
from src.utils.config import load_config
from src.utils.text_utils import obtain_blip_features, obtain_description_embedding
from src.utils.model_utils import setup_model

sys.path.append(os.path.join(os.getcwd(), './submodules/k-diffusion'))

import k_diffusion as K
import argparse
import trimesh


@torch.no_grad()
def main(args):
    
    os.makedirs(os.path.join(args.save_path, args.exp_name, 'upsampled_hairstyle'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.exp_name, 'guiding'), exist_ok=True)
   
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    config = load_config(args.config)
    
    # setup model
    model_ema, accelerator = setup_model(config=config, ckpt_path=args.ckpt_path, device=device)
    
    # setup text embedder model
    model_feature_extractor, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    
    # setup dataset
    hairstyle_dataset = Hairstyle(**config['dataset'])

    # setup upsampler
    if args.save_upsampled_hairstyle:
        upsampler = HairstyleUpsampler(**config['upsampler'], resolution=args.upsample_resolution, hairstyle_dataset=hairstyle_dataset)
    
    cross_cond = None
    
    latent_textures = []
    for sample in range(args.n_samples):
        
        noise = torch.randn(1, config['dataset']['desc_size'], config['dataset']['patch_size'], config['dataset']['patch_size']).cuda()
        sigma = torch.tensor([config['model']['sigma_max']], device=device)
        noised_input = noise * sigma

        sigmas = K.sampling.get_sigmas_karras(args.step,  config['model']['sigma_min'], config['model']['sigma_max'], rho=7., device=device)
        
        if args.precomputed_condition:
            print('we use precomputed condition')
            cross_cond = torch.load(args.precomputed_condition)[None]
        else:
            cross_cond = obtain_description_embedding(hairstyle_description=args.hairstyle_description, average_descriptions=args.average_descriptions, model_feature_extractor=model_feature_extractor, txt_processors=txt_processors)

        extra_args = {}
        extra_args['cross_cond'] = cross_cond
        extra_args['cross_cond_zero'] = torch.zeros(cross_cond.shape[0], 1, config['model']['context_dim'], device=cross_cond.device)

        # denoise texture
        x_0 =  sample_euler_ancestral(model_ema, noised_input, sigmas,  extra_args=extra_args, cfg_scale=args.cfg_scale, disable=not accelerator.is_main_process, seed=args.seed) 
        x_0 = accelerator.gather(x_0)

        latent_textures.append(x_0)
        
        # decode texture into strands
        strands = hairstyle_dataset.texture2strands(x_0) 

        # save obtained hairstyle
        if args.save_guiding_strands:
            cols = torch.cat((torch.rand(strands[0].shape[0], 3).unsqueeze(1).repeat(1, 100, 1),  torch.ones(strands[0].shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            _ = trimesh.PointCloud(strands[0].reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(args.save_path, args.exp_name, 'guiding', f'pc_{sample}.ply')) 
        
        # save upsampled hairstyle
        if args.save_upsampled_hairstyle:
            
            upsampled_pc = upsampler(x_0)
            
            cols = torch.cat((torch.rand(upsampled_pc.shape[0], 3).unsqueeze(1).repeat(1, 100, 1),  torch.ones(upsampled_pc.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            _ = trimesh.PointCloud(upsampled_pc.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(args.save_path, args.exp_name, 'upsampled_hairstyle', f'pc_{sample}.ply'))
            
    if args.save_latent_textures:
        torch.save(torch.stack(latent_textures), os.path.join(args.save_path, args.exp_name, 'exp_textures.pt'))
    
    print('For results see folder ', os.path.join(args.save_path, args.exp_name))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--ckpt_path', default='./data/haar_diffusion.pth', type=str)
    parser.add_argument('--config', default='./configs/train.json', type=str)
    parser.add_argument('--exp_name', default='haar', type=str)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--cfg_scale', default=1.2, type=float, help ='classifier free guidance weight')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_samples', default=10, type=int, help ='number of samples')
    parser.add_argument('--save_path', default='./inference_results', type=str)
    parser.add_argument('--hairstyle_description', default='a woman with straight long hairstyle', type=str)
    
    parser.add_argument('--precomputed_condition', default='', type=str, help ='for using precomputed text embedings')
    parser.add_argument('--average_descriptions', default=True, action="store_true", help ='average descriptions from frontal and back views')    
    parser.add_argument('--save_guiding_strands', default=True, action="store_true", help ='save guiding strands')
    parser.add_argument('--save_upsampled_hairstyle', default=False, action="store_true", help ='save full hairstyle')
    parser.add_argument('--save_latent_textures', default=False, action="store_true", help ='save latent representation of guiding strands')
    parser.add_argument('--upsample_resolution', default=128, type=int, choices=[64, 128, 256], help ='resolution for upsampling hairstyles')
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)