import os
import numpy as np
import trimesh
import sys

from pytorch3d.ops import knn_points

import math
import random

import torch

sys.path.append(os.getcwd())
from src.datasets.dataset import Hairstyle    


class AugmentHairstyle(Hairstyle):
    def __init__(self,
                device='cuda',
                path_to_data='./dataset/pointclouds',
                scalp_path="./data/final_scalp.obj",
                uv_path="./data/new_scalp_uvcoords.pth",
                texture_size=256,
                simple_augments=False,
                curly_augments=False,
                rotation_augments=False,
                cut_augments=False,
                max_curl_radius=0.03,
                max_curls=15,
                min_curl_radius=0.01,
                min_curls=4,
                root_affinity_points=20,
            ):

        self.device = device
        self.scalp_path = scalp_path
        self.uv_path = uv_path

        print('Number of hairstyles is: ', len(os.listdir(path_to_data)))
        self.path_to_data = path_to_data
        self.hairstyle_names = sorted(os.listdir(path_to_data))
        self.n_hairstyles = len(os.listdir(path_to_data))

        #   parameters
        self.texture_size = texture_size
        
        super()._setup_scalp_mesh()
        print('setuped scalp mesh')
        super()._setup_basis()    
        
        print('setuped basis')
        
        super()._setup_meshgrid()
        print('setuped meshgrid')
        
        self.root_affinity_points = root_affinity_points
        self.simple_augments = simple_augments
        self.rotation_augments = rotation_augments
        self.cut_augments = cut_augments
        self.curly_augments = curly_augments


        self.max_curl_radius = max_curl_radius
        self.max_curls = max_curls
        
        self.min_curl_radius = min_curl_radius
        self.min_curls = min_curls

    
    def __len__(self):
        
        return len(os.listdir(self.path_to_data))
    
    
    def _setup_hairstyle(self, idx):  
        
        hairstyle = torch.tensor(trimesh.load(os.path.join(self.path_to_data, self.hairstyle_names[idx])).vertices).reshape(-1, 100, 3).float().to(self.device)

        # if hairstyle has a lot of strands, then subsample additionally it
        nstrands = hairstyle.shape[0]
        if nstrands > 35000:
            a = np.arange(nstrands)
            random.shuffle(a)
            hairstyle = hairstyle[a[:35000]]

        x, y = torch.where(self.meshgrid_mask !=  0)       
        d = knn_points(self.coords_for_each_origin[x, y][None], hairstyle[:, 0][None], K=1)
        meshgrid_verts = hairstyle[:, 0][d[1][0]].squeeze(1)
        shifts = self.coords_for_each_origin[x, y] - meshgrid_verts
        interpolated_hairstyle = hairstyle[d[1][0]].squeeze(1) + shifts.unsqueeze(1)
        self.hair_from_scalp = interpolated_hairstyle
        hairstyle_texture = -100 * torch.ones(self.texture_size, self.texture_size, 100, 3).cuda()
        hairstyle_texture[x, y] = interpolated_hairstyle
        
        return hairstyle_texture #[texture_size, texture_size, 100, 3]
    
    
    def augment_hairstyle(self, hairstyle_texture):

        sampled_grid = self.meshgrid_mask.reshape(-1)                                                     
        masking = torch.nonzero(sampled_grid == 1).reshape(-1)

        sampled_face = self.faces_for_each_origin.reshape(-1)[masking]
        sampled_strands = hairstyle_texture.reshape(-1, 100, 3)[masking]    
        world_strands_aug = self.augment_patch_strands(sampled_face, sampled_strands)

        return world_strands_aug

    
    def augment_patch_strands(self, sampled_face, sampled_strands):
        
#        obtain common patch basis   
        sampled_G = self.R[sampled_face].mean(dim=0, keepdim=True).repeat(sampled_face.shape[0], 1, 1, 1)
        sampled_G_inv = torch.linalg.inv(sampled_G)
        
#        shift all strands such way that origins be (0, 0, 0)
        origins = sampled_strands[:, 0]

        strands_shifted = sampled_strands - origins.unsqueeze(1)

#        switch to patch basis
        patch_strands = (sampled_G @ strands_shifted[..., None])[:, :, :3, 0]
        
#       do squeeze/stretching augmentations
        if self.simple_augments:
            stretch = torch.randn(1, 1, 3).abs() * 0.15 + 1.0
            squeeze = torch.rand(1, 1, 3) < 0.3
            stretch[squeeze] = 1 / stretch[squeeze]
            patch_strands *= stretch.repeat(patch_strands.shape[0], 1, 1).cuda()
            
#        do rotation augmentations     
        if self.rotation_augments:  
            theta = torch.rand(1) * 2 * math.pi
            sin = torch.sin(theta)
            cos = torch.cos(theta)
            Rot = torch.eye(3, device='cuda')[None].expand(1, -1, -1).clone()
            Rot[:, :2, :2] = torch.stack([torch.stack([cos, -sin], dim=-1),
                                    torch.stack([sin, cos], dim=-1)], dim=1)
            patch_strands = (Rot[:, None] @ patch_strands[..., None])[..., 0]

#        do curly augmentations
        value = random.random()
        if self.curly_augments and patch_strands.shape[0] != 0:
            eps = 1e-4
            t = torch.empty_like(patch_strands)
            t[:, :-1, :] = patch_strands[:, 1:, :] - patch_strands[:, :-1, :]
            t[:, -1] = t[:, -2]

            t = self.smooth(a=t, n=32)
            t = t / (torch.linalg.norm(t, axis=2, keepdims=True) + eps)

            b = torch.empty_like(t)
            b[:, 1:, :] = torch.cross(t[:, :-1, :], t[:, 1:, :])
            b[:, 0] = b[:, 1]
            b = self.smooth(a=b, n=32)
            b = b / (torch.linalg.norm(b, axis=2, keepdims=True) + eps)

            n = torch.cross(t, b)
            n = self.smooth(a=n, n=32)
            n = n / (torch.linalg.norm(n, axis=2, keepdims=True) + eps)

            
            i = torch.linspace(0, self.max_curls * np.pi* random.random(), 100, device=self.device).repeat(len(patch_strands), 1).unsqueeze(-1)

            r = random.random() * self.max_curl_radius * torch.cat((
                                    torch.sin(torch.linspace(0, np.pi / 2, self.root_affinity_points, device=self.device).reshape(-1, 1)),
                                    torch.ones(100 - self.root_affinity_points, 1, device=self.device)))[None].repeat(len(patch_strands), 1, 1)
           
            patch_strands = patch_strands + r * (torch.cos(i) * b + torch.sin(i) * n)

#        do cutting augmentations        
        value = random.random()
        if self.cut_augments and value > 0.7:

            patch_strands = torch.nn.functional.interpolate(
                patch_strands[: , : 50 + int(random.random() * 50), :].permute(0, 2, 1),
                100,
                align_corners=True,
                mode='linear').permute(0, 2, 1)
        
#        return to world basis
        world_strands = (sampled_G_inv @ patch_strands[..., None])[:, :, :3, 0] + origins.unsqueeze(1)
        return  world_strands
    
    
    def smooth(self, a, n=4, pad=True):
        if pad:

            ret = torch.cat((torch.repeat_interleave(a[:, :1, :], n - 1, dim=1), a), dim=1)
            ret = torch.cumsum(ret, dim=1, dtype=a.dtype)
        else:
            ret = torch.cumsum(a, dim=1, dtype=a.dtype)
    
        ret[:, n: ] = ret[:, n: ] - ret[:, : -n]
        
        return ret[:, n - 1: ] / n
    

    def __getitem__(self, idx): 

        hairstyle_texture = self._setup_hairstyle(idx)
        world_str_aug = self.augment_hairstyle(hairstyle_texture)
        nstrands = world_str_aug.shape[0]

        if nstrands > 20000:
            a = np.arange(nstrands)
            random.shuffle(a)
            world_str_aug = world_str_aug[a[:20000]]

        return world_str_aug
            