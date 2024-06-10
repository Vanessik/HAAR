from PIL import Image
import os
import numpy as np
import sys
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), './submodules/LLaVA'))
sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))

from lavis.models import load_model_and_preprocess
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

import argparse

from src.utils.text_utils import QUESTIONS, obtain_blip_features
from preprocess_dataset.obtain_hairstyle_descriptions import obtain_hairstyle_description


@torch.no_grad()
def main(args):
    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
    image = Image.open(args.path_to_image)
    
    # obtain hairstyle description
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)

    hairstyle_description = obtain_hairstyle_description(
                                    image, 
                                    tokenizer,
                                    model,
                                    image_processor,
                                    context_len,
                                    questions=QUESTIONS,
                                    temperature=0.1
                                       )

    torch.cuda.empty_cache()
    
    # obtain text embedding
    model_feature_extractor, _, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    description = [desc for desc in hairstyle_description if 'i cannot' not in desc and 'i am not sure' not in desc]
    embs = [obtain_blip_features(sentence_description, model_feature_extractor, txt_processors).mean(0) for sentence_description in description]
    condition = torch.stack(embs).mean(0, keepdim=True)
    
    torch.save(condition, args.save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-13b', type=str)
    parser.add_argument('--save_path', default='./data/precomputed_condition.pt', type=str)
    parser.add_argument('--path_to_image', default='/is/rg/ncs/projects/vsklyarova/neuralhaircut/data/monocular/jenya/image/img_0000.png', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)