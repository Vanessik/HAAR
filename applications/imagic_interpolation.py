import os
import sys
import torch
from tqdm.auto import tqdm
import argparse
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))
from lavis.models import load_model_and_preprocess

from src.datasets.dataset import Hairstyle
from src.upsampling.upsampler import HairstyleUpsampler

from src.sampler import sample_euler_ancestral
from src.utils.config import load_config

from src.utils.text_utils import obtain_blip_features
from src.utils.model_utils import setup_model, setup_train_model

sys.path.append(os.path.join(os.getcwd(), './submodules/k-diffusion'))

import k_diffusion as K
import argparse
import trimesh


@torch.no_grad()
def inference_interpolation(args, model_ema, accelerator, hairstyle_dataset, upsampler, emb_hairstyle_1, emb_hairstyle_2, config, device, stage):
    
    os.makedirs(os.path.join(args.save_path, args.exp_name, stage, 'upsampled_hairstyle'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.exp_name, stage, 'guiding'), exist_ok=True)
    
    weights = torch.linspace(0, 1, args.n_interpolation_states).cuda() 
   
    noise = torch.randn(1, config['dataset']['desc_size'], config['dataset']['patch_size'], config['dataset']['patch_size']).cuda()
    sigma = torch.tensor([args.degree], device=device)
    noised_input = noise * sigma

    sigmas = K.sampling.get_sigmas_karras(args.step,  config['model']['sigma_min'], args.degree, rho=7., device=device)
 
    latent_textures = []
    for sample in range(weights.shape[0]):
        
        # interpolate embeddings of hairstyle
        cross_cond = weights[sample] * emb_hairstyle_1 + (1 - weights[sample]) * emb_hairstyle_2
        
        extra_args = {}
        extra_args['cross_cond'] = cross_cond
        extra_args['cross_cond_zero'] =  torch.zeros(cross_cond.shape[0], 1, config['model']['context_dim'], device=cross_cond.device)

        # denoise texture
        x_0 =  sample_euler_ancestral(model_ema, noised_input, sigmas,  extra_args=extra_args, cfg_scale=args.cfg_scale, disable=not accelerator.is_main_process, seed=args.seed) 
        x_0 = accelerator.gather(x_0)

        latent_textures.append(x_0)
        
        
        # decode texture into strands
        strands = hairstyle_dataset.texture2strands(x_0) 

        
        # save obtained hairstyle
        if args.save_guiding_strands:
            cols = torch.cat((torch.rand(strands[0].shape[0], 3).unsqueeze(1).repeat(1, 100, 1),  torch.ones(strands[0].shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            _ = trimesh.PointCloud(strands[0].reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(args.save_path, args.exp_name, stage, 'guiding', f'pc_{sample}.ply')) 
            
            
         # save upsampled hairstyle 
        if args.save_upsampled_hairstyle:
            upsampled_pc = upsampler(x_0)
            
            cols = torch.cat((torch.rand(upsampled_pc.shape[0], 3).unsqueeze(1).repeat(1, 100, 1),  torch.ones(upsampled_pc.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            _ = trimesh.PointCloud(upsampled_pc.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(args.save_path, args.exp_name, stage, 'upsampled_hairstyle', f'pc_{sample}.ply'))
            

    if args.save_latent_textures:
        torch.save(torch.stack(latent_textures), os.path.join(args.save_path, args.exp_name, stage, 'interpolation_textures.pt'))


        
def main(args):
    
    torch.manual_seed(args.seed)
      
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # setup config
    config = load_config(args.config)
    
    # setup model
    model, accelerator = setup_model(config=config, ckpt_path=args.ckpt_path, device=device)
    
    # setup text embedder model
    model_feature_extractor, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    
    # setup dataset
    hairstyle_dataset = Hairstyle(**config['dataset'])
    print('Setup hairstyle dataloader')
    
    # setup upsampler
    if args.save_upsampled_hairstyle:
        upsampler = HairstyleUpsampler(**config['upsampler'], resolution=args.upsample_resolution, hairstyle_dataset=hairstyle_dataset)
    
    # upload target hairstyle from which we want interpolate
    input_texture = torch.load(args.target_hairstyle_texture)[0] #Should have resolution of [1, 64, patch_size, patch_size]

    # upload prompt with which to edit it
    e_tgt = obtain_blip_features(args.hairstyle_edit_prompt, model_feature_extractor, txt_processors).cuda().mean(0, keepdim=True)[None]

    sample_density = K.config.make_sample_density(config["model"])
        
    # Stage 1. Text embedding optimization
    
    e_opt = e_tgt.clone()
    e_opt.requires_grad = True
    
    opt = torch.optim.Adam([e_opt], lr=args.lr_stage_1)

    for i in tqdm(range(args.n_iter_stage_1)):
        opt.zero_grad()

        noise = torch.randn(1, config['dataset']['desc_size'], config['dataset']['patch_size'], config['dataset']['patch_size']).cuda()     
        sigma = sample_density([noise.shape[0]], device=device)
        losses = model.loss(input_texture, noise, sigma, mask=hairstyle_dataset.average_mask, cross_cond=e_opt)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.)

        loss = accelerator.gather(losses).mean().item()

        accelerator.backward(losses.mean())
        opt.step()
    

    opt_emb = e_opt.clone()
    inference_interpolation(args, model, accelerator, hairstyle_dataset, upsampler, e_tgt, opt_emb, config, device, stage='stage1')
    
    print('Finish first stage of Imagic with text embedding optimization')
    
    
    # Stage 2. Model finetuning
    cross_cond = opt_emb.clone()
    reals = input_texture.clone()
    mask = hairstyle_dataset.average_mask.clone()

    opt_config = config['optimizer']
    model_config = config['model']
    
    ema_stats = {}

    inner_model, accelerator, model, ema_sched, model_ema = setup_train_model(config, args.ckpt_path, device)

    params = list(inner_model.parameters()) 
    opt = torch.optim.AdamW(params,
                      lr=args.lr_stage_2,
                      betas=tuple(opt_config['betas']),
                      eps=opt_config['eps'],
                      weight_decay=opt_config['weight_decay'])
    

    for i in tqdm(range(args.n_iter_stage_2)):
        opt.zero_grad()
        noise = torch.randn_like(reals)
        sigma = sample_density([reals.shape[0]], device=device)
        losses = model.loss(reals, noise, sigma, mask=mask, cross_cond=cross_cond)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.)

        loss = accelerator.gather(losses).mean().item()

        accelerator.backward(losses.mean())

        opt.step()
        ema_decay = ema_sched.get_value()
        K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay )
        if accelerator.sync_gradients:
            K.utils.ema_update(model, model_ema, ema_decay)
            ema_sched.step()

    inference_interpolation(args, model_ema.eval(), accelerator, hairstyle_dataset, upsampler, e_tgt, opt_emb, config, device, stage='stage2')
    
    print('Finish second stage of Imagic with text embedding optimization')
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--ckpt_path', default='./data/haar_diffusion.pth', type=str)
    parser.add_argument('--config', default='./configs/config.json', type=str)
    parser.add_argument('--exp_name', default='haar', type=str)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--degree', default=80, type=int)
    parser.add_argument('--cfg_scale', default=1.5, type=float)
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--n_interpolation_states', default=10, type=int)
    
    parser.add_argument('--save_path', default='./inference_results', type=str)
    
    parser.add_argument('--lr_stage_1', default=0.001, type=float)
    parser.add_argument('--n_iter_stage_1', default=1500, type=int)
    
    parser.add_argument('--lr_stage_2', default=0.0001, type=float)
    parser.add_argument('--n_iter_stage_2', default=600, type=int)
    
    parser.add_argument('--hairstyle_edit_prompt', default='short hairstyle', type=str)
    parser.add_argument('--target_hairstyle_texture', default='exp_texture.pt', type=str)

    parser.add_argument('--save_guiding_strands', default=True, action="store_true", help ='save guiding strands')
    parser.add_argument('--save_upsampled_hairstyle', default=False, action="store_true", help ='save full hairstyle')
    parser.add_argument('--save_latent_textures', default=False, action="store_true", help ='save latent representation of guiding strands')
    parser.add_argument('--upsample_resolution', default=128, type=int, help ='resolution for upsampling hairstyles')

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)