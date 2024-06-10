eval "$(conda shell.bash hook)"


source /home/vsklyarova/miniconda3/bin/activate

conda activate haar && python applications/image2hairstyle.py --save_path ./data/precomputed_condition.pt --path_to_image ./examples/example1.png

python infer.py --exp_name example1 --ckpt_path ./data/haar_diffusion.pth --conf ./configs/infer.json --cfg_scale 1.5 --precomputed_condition ./data/precomputed_condition.pt --n_samples 10 --save_latent_textures --save_guiding_strands --save_upsampled_hairstyle --upsample_resolution 64