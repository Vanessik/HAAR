import os
import numpy as np
import trimesh
import sys

from pytorch3d.ops import  knn_points
from pytorch3d.io import load_objs_as_meshes


import torch
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), './submodules/NeuralHaircut/src/hair_networks'))

from strand_prior import Encoder, Decoder
from src.utils.geometry import map_uv_to_3d, barycentric_coordinates_of_projection


class Hairstyle(Dataset):
    def __init__(self,
                 device='cuda',
                 path_to_data='./dataset/pointclouds',
                 scalp_path="./data/final_scalp.obj",
                 uv_path="./data/symmetry_scalp_uvcoords.pth",
                 enc_ckpt="./pretrained_models/strand_prior/strand_ckpt.pth",
                 feats_path='./dataset/features/',
                 texture_size=64,
                 patch_size=32,
                 desc_size=64,
                 dec_length=99,
                 num_classes=0, #for compatibility with k-diffusion config, todo remove it
                 type = ''  
                ):
        
        
        self.feats_path = feats_path

        self.device = device
        self.scalp_path = scalp_path
        self.uv_path = uv_path

        print('Number of hairstyles is: ', len(os.listdir(path_to_data)))
        self.path_to_data = path_to_data
        self.n_hairstyles = len(os.listdir(path_to_data))
        
        self.hairstyles_dir = sorted(os.listdir(path_to_data))

        #   parameters
        self.texture_size = texture_size
        self.patch_size = patch_size
        self.desc_size = desc_size
        
        self._setup_scalp_mesh()
        print('Finish setupping scalp mesh!')
        self._setup_basis()    
        
        print('Finish setupping basis!')
        
        self._setup_meshgrid()
        print('Finish setupping meshgrid!')
        
#         setup encoder to produce latents
        self.enc = Encoder(None, latent_dim=desc_size).eval().cuda()

        ckpt = torch.load(enc_ckpt, map_location='cpu')
        self.enc.load_state_dict(ckpt['encoder'])
        
        self.dec = Decoder(None, latent_dim=desc_size, length=dec_length).eval().cuda()
        self.dec.load_state_dict(ckpt['decoder'])

        self._create_average_texture()


    def _create_average_texture(self):
        
        '''Create average texture for hairstyle calculation'''

        masks = []
    
        for _ in range(1000):
            idx, idy = Hairstyle.downsample_texture(texture_size=self.texture_size, patch_size=self.patch_size, device=self.device)

            masks.append(self.meshgrid_mask[[idx, idy]].reshape(self.patch_size, self.patch_size))

        average_mask = torch.stack(masks).sum(0) / 1000
        average_mask = average_mask == 1

        small_faces = self.faces_for_each_origin[[idx, idy]].reshape(self.patch_size, self.patch_size)
        small_origins = self.coords_for_each_origin[[idx, idy]].reshape(self.patch_size, self.patch_size, 3)

        self.nonzerox,  self.nonzeroy = torch.where(average_mask != 0)
        
        self.average_mask = average_mask

        self.small_R_inv = self.R_inv[small_faces[self.nonzerox, self.nonzeroy]][:, None]
        self.small_origins = small_origins[self.nonzerox, self.nonzeroy].unsqueeze(1)


    def texture2strands(self, texture):
        '''
        texture = [bs, 64, self.patch_size, self.patch_size]
        '''
        
        bs = texture.shape[0]
        decoded = self.dec(texture.permute(2, 3, 0, 1)[[self.nonzerox, self.nonzeroy]].permute(1, 0, 2).reshape(-1, 64)).reshape(bs, -1, 99, 3)
        

        p_local = torch.cat([
                                torch.zeros_like(decoded[:, :, -1:, :]), 
                                torch.cumsum(decoded, dim=2)
                            ], 
                            dim=2
                        )
        

        p_world = (self.small_R_inv[None].repeat(bs, 1, 1, 1, 1) @  p_local[..., None])[:, :, :, :3, 0] + self.small_origins[None].repeat(bs, 1, 1, 1)
    
        return p_world
        
        
    def _setup_meshgrid(self):
        
        u_values = np.linspace(-1., 1., self.texture_size)
        v_values = np.linspace(-1., 1., self.texture_size)
        U_np, V_np = np.meshgrid(u_values, v_values)

        # Map 2D UV meshgrid points to 3D positions and face index
        positions_face_result_torch =  map_uv_to_3d(U_np, V_np, self.scalp_mesh.faces_packed().cpu().numpy(), self.scalp_uvs.cpu().numpy(), self.scalp_mesh.verts_packed().cpu().numpy())
        
        self.faces_for_each_origin = positions_face_result_torch[:, :, 3].long().cuda()  # [texture_size, texture_size]
        self.coords_for_each_origin = positions_face_result_torch[:, :, :3].float().cuda()  # [texture_size, texture_size, 3]
        self.meshgrid_mask = (self.faces_for_each_origin != -100).bool().cuda()  # [texture_size, texture_size]
              
            
    def _setup_scalp_mesh(self):
        
        self.scalp_uvs = torch.load(self.uv_path)[None].float().to(self.device) 
        self.scalp_mesh = load_objs_as_meshes([self.scalp_path], device=self.device)

        
    def _setup_basis(self):
        
        '''Create scalp basis'''
        
        full_uvs = self.scalp_uvs[0][self.scalp_mesh.faces_packed()]
        bs = full_uvs.shape[0]
        concat_full_uvs = torch.cat((full_uvs, torch.zeros(bs, full_uvs.shape[1], 1, device=full_uvs.device)), -1)
        new_point = concat_full_uvs.mean(1).clone()
        new_point[:, 0] += 0.01 
        bary_coords = barycentric_coordinates_of_projection(new_point, concat_full_uvs).unsqueeze(1)

        full_verts = self.scalp_mesh.verts_packed()[self.scalp_mesh.faces_packed()]
        
        origin_t = (bary_coords @ full_verts).squeeze(1) - full_verts.mean(1)
        origin_t /= origin_t.norm(dim=-1, keepdim=True)

        origin_n = self.scalp_mesh.faces_normals_packed()
        origin_n /= origin_n.norm(dim=-1, keepdim=True)

        origin_b = torch.cross(origin_n, origin_t, dim=-1)
        origin_b /= origin_b.norm(dim=-1, keepdim=True)
        
        self.R = torch.stack([origin_t, origin_b, origin_n], dim=1) # global to local  
        
        # Construct an inverse transform from local to global coords
        self.R_inv = torch.linalg.inv(self.R) 
        
        
    def _setup_data(self, idx):
        
        self._setup_features(idx) 
        return self._setup_hairstyle(idx)

    
    def _setup_features(self, idx):
        
        '''Upload embeddings of textual descriptions for hairstyle idx'''     
        
        frontal_desc = torch.load(os.path.join(self.feats_path, f'{idx:05d}', 'frontal.pt'), map_location='cpu') 
        back_desc =  torch.load(os.path.join(self.feats_path, f'{idx:05d}', 'back.pt'), map_location='cpu') 

        self.features = torch.concat((frontal_desc, back_desc), 0).cuda()
        self.features = self.features.mean(0, keepdim=True)

            
    def __len__(self):
        
        return self.n_hairstyles

    
    def _setup_hairstyle(self, idx):  
        
        hairstyle = torch.tensor(trimesh.load(os.path.join(self.path_to_data, self.hairstyles_dir[idx])).vertices).reshape(-1, 100, 3).float().to(self.device)
  
        x, y = torch.where(self.meshgrid_mask !=  0)       
        d = knn_points(self.coords_for_each_origin[x, y][None], hairstyle[:, 0][None], K=1)
        meshgrid_verts = hairstyle[:, 0][d[1][0]].squeeze(1)
        shifts = self.coords_for_each_origin[x, y] - meshgrid_verts
        interpolated_hairstyle = hairstyle[d[1][0]].squeeze(1) + shifts.unsqueeze(1)
        self.hair_from_scalp = interpolated_hairstyle
        hairstyle_texture = -100 * torch.ones(self.texture_size, self.texture_size, 100, 3).cuda()
        hairstyle_texture[x, y] = interpolated_hairstyle
        
        return hairstyle_texture #[texture_size, texture_size, 100, 3]
    
    
    def downsample_hairstyle(self, hairstyle_patch):
        
        idx, idy = Hairstyle.downsample_texture(texture_size=self.texture_size, patch_size=self.patch_size, device=self.device)

        train_mask_patch = self.meshgrid_mask[[idx, idy]].reshape(self.patch_size, self.patch_size)
        small_faces = self.faces_for_each_origin[[idx, idy]].reshape(self.patch_size, self.patch_size)

        nonzero_idx, nonzero_idy  = torch.where(train_mask_patch !=  0) 
        small_hairstyle = hairstyle_patch[[idx, idy]].reshape(self.patch_size, self.patch_size, 100, 3)
        small_hairstyle = small_hairstyle[[nonzero_idx, nonzero_idy]]
            
        shifted_strands = small_hairstyle - small_hairstyle[:, 0].unsqueeze(1)
        local_strands = (self.R[small_faces[nonzero_idx, nonzero_idy]][:, None] @ shifted_strands[..., None])[:, :, :3, 0] 
        
        with torch.no_grad():
            latent = self.enc(local_strands)[:, :self.desc_size]
            
        train_latent_patch = torch.zeros(self.patch_size, self.patch_size, 64).to(self.device)
        train_latent_patch[[nonzero_idx, nonzero_idy]] = latent
        
        return train_latent_patch, train_mask_patch
        
        
    @staticmethod    
    def downsample_texture(texture_size, patch_size, device='cuda'):
        
        '''Subsample texture'''
        
        b = torch.linspace(0, texture_size ** 2 - 1, texture_size ** 2, device=device).reshape(texture_size, texture_size)

        n_patches = texture_size // patch_size

        unf = torch.nn.Unfold(
                            kernel_size=n_patches,
                            stride=n_patches
                             )
        
        unfo = unf(b[None, None]).reshape(-1, patch_size ** 2)

        idx_ = torch.randint(low=0, high=n_patches ** 2, size=(patch_size ** 2,), device=device)

        choosen_val = unfo[idx_, :].diag()

        x = choosen_val // texture_size
        y = choosen_val % texture_size 
        
        return x.long(), y.long()
        
        
    def __getitem__(self, idx): 
        hairstyle = self._setup_data(idx)
        train_latent_patch, train_mask_patch = self.downsample_hairstyle(hairstyle)

        return train_latent_patch.permute(2, 0, 1), train_mask_patch[None], self.features