import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append(os.getcwd())

sys.path.append(os.path.join(os.getcwd(), './submodules/LLaVA'))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from src.utils.text_utils import QUESTIONS



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def obtain_hairstyle_description(
           image, 
           tokenizer,
           model,
           image_processor,
           context_len,
           questions,
           conv_mode='llava_v1',
           temperature=0.1
           ):

    answers = []
    
    for idx, qs in enumerate(questions):    
        
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_id = shortuuid.uuid()
        bad_chars = ['\"']

        outputs = ''.join((filter(lambda i: i not in bad_chars, outputs)))

        outputs = outputs[0].lower() + outputs[1:]
        answers += [outputs]
        
    return answers



def llava_eval_model(
               tokenizer,
               model,
               image_processor,
               context_len,
               questions,
               answers_file, 
               img_path,
               conv_mode='llava_v1',
               temperature=0.1,
               pc_idx=0
               ):
    
    os.makedirs(answers_file, exist_ok=True)
    
    ans_file = open(os.path.join(answers_file, f'{pc_idx:05d}.txt'), "w")
    
    
    for idx, qs in enumerate(questions):    
        
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        
        output_views = []
        for view_idx, view in enumerate(['frontal', 'back']):
            image = Image.open(os.path.join(img_path, f'{pc_idx:05d}', 'image', f'img_{view_idx:04d}.png'))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()

            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            bad_chars = ['\"']


            outputs = ''.join((filter(lambda i: i not in bad_chars,
                                  outputs)))

            outputs = f'From {view} view ' + outputs[0].lower() + outputs[1:]

            output_views.append(outputs)

        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text_frontal": output_views[0],
                                   "text_back": output_views[1],
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    


def main(args):
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    
    num_hairstyles = len(os.listdir(args.image_path))
    
    for i in tqdm(range(num_hairstyles)):
        print(f'Start processing {i}')
        
        llava_eval_model(tokenizer,
                   model,
                   image_processor,
                   context_len,
                   questions=QUESTIONS,
                   pc_idx=i,
                   answers_file=args.answers_file_path,
                   img_path = args.image_path,
                   temperature=args.temperature
                  )
        
    torch.cuda.empty_cache()


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--model_path', default='liuhaotian/llava-v1.5-13b', type=str)
    
    parser.add_argument('--answers_file_path', default='./dataset/answers/', type=str)
    parser.add_argument('--image_path', default='./dataset/blender/', type=str)
    parser.add_argument('--temperature', default=0.1, type=float)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)  