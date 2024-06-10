import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
from src.upsampling.utils import calc_strands_similarity


class HairstyleUpsampler(nn.Module):
    def __init__(self,
                 mode='mixed',
                 use_noise=True,
                 noise_mean=0.15,
                 noise_std=0.05,
                 path_to_coords='./data/coords_for_each_origin_512.pt',
                 path_to_faces='./data/faces_for_each_origin_512.pt',
                 path_to_meshgrid='/fast/vsklyarova/Projects/GenHair/k-diffusion/meshgrid_mask3.pt',
                 path_to_basis='./data/R_inv_512.pt',
                 device='cuda',
                 resolution=128,
                 hairstyle_dataset=None,
                ):
        
        super().__init__()
               
        self.mode = mode
        self.resolution = resolution
        self.use_noise = use_noise

        self.coords_for_each_origin_512  = torch.load(path_to_coords)
        self.faces_for_each_origin_512 = torch.load(path_to_faces)
        self.meshgrid_mask_512 = torch.load(path_to_meshgrid)
        self.R_inv_512 = torch.load(path_to_basis)
        
        self.device = device
        self.use_noise = use_noise

        if self.mode == 'mixed':
            self.blend_func = lambda x: torch.where(x <= 0.9, 1 - 1.63 * x ** 5, 0.4 - 0.4 * x)
            
            if self.use_noise:
                self.noise_distribution = Normal(torch.tensor([noise_mean], device=self.device), torch.tensor([noise_std], device=self.device))
                self.sample_ones = torch.tensor(np.random.choice([-1, 1], size=self.resolution * self.resolution).reshape(self.resolution, self.resolution), device=self.device)[None]
        
        self.hairstyle_dataset = hairstyle_dataset
        
        self._calc_nonzero_mask()

        
    @torch.no_grad()
    def upsample_latents(self, latents, decoder):
        
        n = latents.shape[0] // 1000
        strands_list = []

        for i in range(n + 1):

            l, r = i * 1000, (i+1) * 1000

            z_geom_batch = latents[l:r]

            v = decoder(z_geom_batch)
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )

            world_strands = (self.R_inv_512[self.resolution_faces[self.nonzerox, self.nonzeroy]][l:r, None] @  p_local[..., None])[:, :, :3, 0] + self.resolution_origins[self.nonzerox, self.nonzeroy][l:r].unsqueeze(1)

            strands_list.append(world_strands)

        return torch.cat(strands_list, dim=0)
    

    def _calc_nonzero_mask(self):
        
        if self.resolution != 512:

            idx, idy = self.hairstyle_dataset.downsample_texture(texture_size=512, patch_size=self.resolution)
            
            small_mask = self.meshgrid_mask_512[[idx, idy]].reshape(self.resolution, self.resolution)
            self.resolution_faces = self.faces_for_each_origin_512[[idx, idy]].reshape(self.resolution, self.resolution)
            self.resolution_origins = self.coords_for_each_origin_512[[idx, idy]].reshape(self.resolution, self.resolution, 3)
            
        else:

            small_mask = self.meshgrid_mask_512.reshape(self.resolution, self.resolution)
            self.resolution_faces = self.faces_for_each_origin_512.reshape(self.resolution, self.resolution)
            self.resolution_origins = self.coords_for_each_origin_512.reshape(self.resolution, self.resolution, 3)
        
        self.nonzerox, self.nonzeroy = torch.where(small_mask != 0)

        

    def forward(self, texture):
        
        '''
        Upsample initial sparse guiding strands to desired resolution
        
        Input:
        texture = [1, 64, patch_size, patch_size]

        Return:
        [N_strands, 100, 3]
        
        '''

        bil = F.interpolate(texture, self.resolution, mode='bilinear')[0]
        latents_interp = bil.permute(1, 2, 0)[[self.nonzerox, self.nonzeroy]].reshape(-1, 64)

        if self.mode == 'bilinear':
            return self.upsample_latents(latents_interp, self.hairstyle_dataset.dec)
            
        near = F.interpolate(texture, self.resolution, mode='nearest')[0] 
        latents_interp = near.permute(1, 2, 0)[[self.nonzerox, self.nonzeroy]].reshape(-1, 64)

        if self.mode == 'nearest':
            return self.upsample_latents(latents_interp, self.hairstyle_dataset.dec)

            
        if self.mode == 'mixed':
            
            strands_origins = self.hairstyle_dataset.texture2strands(texture)[0]

            patch_world_displ = torch.zeros(self.hairstyle_dataset.patch_size, self.hairstyle_dataset.patch_size, 99, 3, device=self.device)
            patch_world_displ[[self.hairstyle_dataset.nonzerox, self.hairstyle_dataset.nonzeroy]] = strands_origins[:, 1:] - strands_origins[:, :-1]

            strands_sim = calc_strands_similarity(patch_world_displ)
            strands_sim_hr = F.interpolate(strands_sim[None][None], self.resolution, mode='bilinear')[0][0]

            latents_interp = self.blend_func(strands_sim_hr)[None] * near + (1 - self.blend_func(strands_sim_hr)[None]) * bil

            if self.use_noise:

                lat_std = latents_interp.permute(1, 2, 0)[[self.nonzerox, self.nonzeroy]].reshape(-1, 64).std(0).unsqueeze(-1).unsqueeze(-1)
                latents_interp += (lat_std * (self.sample_ones * self.noise_distribution.sample(sample_shape=torch.tensor([self.resolution, self.resolution], device=self.device)).permute(2, 0, 1))).float()
                
            latents = latents_interp.permute(1, 2, 0)[[self.nonzerox, self.nonzeroy]].reshape(-1, 64)
            
            return self.upsample_latents(latents, self.hairstyle_dataset.dec)