import os
import torch

import trimesh
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from src.datasets.aug_dataset import AugmentHairstyle
import argparse


def main(args):
              
    aug_hairstyle = AugmentHairstyle(device=args.device,
                path_to_data=args.path_to_data,
                scalp_path=args.scalp_path,
                uv_path=args.uv_path,
                texture_size=args.texture_size,
                simple_augments=args.simple_augments,
                curly_augments=args.curly_augments,
                rotation_augments=args.rotation_augments,
                cut_augments=args.cut_augments,
                max_curl_radius=args.max_curl_radius,
                max_curls=args.max_curls,
                min_curl_radius=args.min_curl_radius,
                min_curls=args.min_curls,
                root_affinity_points=args.root_affinity_points,
            )

    save_path = args.save_path
    
    os.makedirs(save_path, exist_ok=True)
    
    for sample in tqdm(range(aug_hairstyle.__len__())):

        for j in range(args.num_variations):

            idx = args.num_variations * sample + j

            strands = aug_hairstyle.__getitem__(sample)

            cols = torch.cat((torch.rand(strands.shape[0], 1, 3).repeat(1, 100, 1), torch.ones(strands.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()
            _ = trimesh.PointCloud(strands.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(save_path, f'pc_{idx:05d}.ply'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--path_to_data', default='./dataset/pointclouds', type=str)
    parser.add_argument('--save_path', default='./dataset/augmented_pointclouds', type=str)
    parser.add_argument('--uv_path', default='./data/symmetry_scalp_uvcoords.pth', type=str)
    parser.add_argument('--scalp_path', default='./data/final_scalp.obj', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_variations', default=25, type=int)
    parser.add_argument('--texture_size', default=256, type=int)
    parser.add_argument('--simple_augments', default=False, action="store_true", help ='add stretching/squeezing augmentations')
    parser.add_argument('--rotation_augments', default=False, action="store_true", help ='add rotation augmentations')
    parser.add_argument('--curly_augments', default=False, action="store_true", help ='add curly augmentations')
    parser.add_argument('--cut_augments', default=False, action="store_true", help ='add cutting augmentations')
    parser.add_argument('--max_curl_radius', default=0.0025, type=float)
    parser.add_argument('--max_curls', default=8, type=int)
    parser.add_argument('--min_curl_radius', default=0.002, type=int)
    parser.add_argument('--min_curls', default=5, type=int)
    parser.add_argument('--root_affinity_points', default=20, type=int)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)