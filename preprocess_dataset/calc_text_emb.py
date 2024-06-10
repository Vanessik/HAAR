import json
import torch
import sys
import os

sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))
from lavis.models import load_model_and_preprocess

import torch

from tqdm import tqdm
import random

sys.path.append(os.getcwd())
from src.utils.text_utils import ALL_QUESTIONS, GENERAL, FRONT, BACK, obtain_blip_features

import argparse


MAPPING = {
       'frontal': FRONT,
       'back': BACK
      }


def main(args):
    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model_feature_extractor, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)

    dataset = sorted(os.listdir(args.text_path))
   
    save_path = args.save_path

    all_embs_frontal = []
    all_embs_back = []

    num_hairstyles = len(dataset)
    for i in tqdm(range(num_hairstyles)):
        
        print(dataset[i].split('.')[0])
        idx_name = dataset[i].split('.')[0] 

        os.makedirs(os.path.join(args.save_path, f'{idx_name}'), exist_ok=True)

        f = open(os.path.join(args.text_path, dataset[i]), "r")
        d = []
        for _ in range(ALL_QUESTIONS):
            string = f.readline()
            description = json.loads(string)
            d.append(description)

        questions = {}
        
        for view in ['frontal', 'back']:
            not_reliable = ['i cannot' in l[f'text_{view}'] for l in d]
            r_important = [r for r in MAPPING[view] if not_reliable[r] is False]
            r_sampled = [r for r in GENERAL if not_reliable[r] is False]
            final = sorted(r_important + random.sample(r_sampled, min(2, len(r_sampled))))
            questions[view] = final

        embs_frontal = []
        embs_back = []

        for sample in range(ALL_QUESTIONS):
            
            if sample in questions['frontal']:
                embs_frontal.append(obtain_blip_features(d[sample]['text_frontal'], model_feature_extractor, txt_processors).mean(0))

            if sample in questions['back']:
                embs_back.append(obtain_blip_features(d[sample]['text_back'], model_feature_extractor, txt_processors).mean(0))

        all_embs_frontal.append(torch.stack(embs_frontal).mean(0, keepdim=True).cpu())
        all_embs_back.append(torch.stack(embs_back).mean(0, keepdim=True).cpu())

        torch.save(torch.stack(embs_frontal).cpu(), os.path.join(save_path, f'{idx_name}', f'frontal.pt'))
        torch.save(torch.stack(embs_back).cpu(), os.path.join(save_path, f'{idx_name}', f'back.pt'))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--save_path', default='./dataset/features/', type=str)
    parser.add_argument('--text_path', default='./dataset/answers', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)