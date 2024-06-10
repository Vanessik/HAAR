#!/bin/bash

conda activate haar && python infer.py --exp_name infer_haar --conf ./configs/infer.json --ckpt_path ./pretrained_models/haar_prior/haar_diffusion.pth --cfg_scale 1.5 --save_latent_textures --save_guiding_strands --save_upsampled_hairstyle --upsample_resolution 64 --n_samples 10 --hairstyle_description "long straight hairstyle"