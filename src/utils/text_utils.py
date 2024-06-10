import torch


FRONT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 21, 22]
BACK = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 21, 22]
GENERAL = [25, 26, 24, 23, 19, 15]
ALL_QUESTIONS = 27

QUESTIONS = [    
   'Describe in detail the bang/fringe of depicted hairstyle including its directionality, texture and coverage of face? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'what is the overall hairstyle depicted in the image? If you are not sure say it honestly. Do not imagine any contents that is no in the image.  After answer please clear you history.',
    'Does the depicted hairstyle longer than the shoulders or shorter than the shoulder? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    "Does the depicted hairstyle has short bang or long bang or no bang from frontal view? If you are not sure say it honestly. Do not imagine any contents that is no in the image.  After answer please clear you history.",
    "Does the hairstyle has straight bang or Baby Bangs or Arched Bangs or Asymmetrical Bangs or Pin-Up Bangs or Choppy Bangs or curtain bang or side swept bang or no bang? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.",
    "Are there any afro features in the hairstyle or no afro features? If you are not sure say it honestly. Do not imagine any contents that is no in the image.  After answer please clear you history.",
    "Is the length of hairstyle shorter than middle of the neck or longer than middle of the neck?If you are not sure say it honestly. Do not imagine any contents that is no in the image.  After answer please clear you history.",
    'What is the main geometry features of the depicted hairstyle? If you are not sure say it honestly. Do not imagine any contents that is no in the image.  After answer please clear you history.',
    'What is the overall shape of the depicted hairstyle? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Is the hair short, medium, or long in terms of length? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.', 
    'What is the type of depicted hairstyle? If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'What is the length of hairstyle relative to human body? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    "Describe the texture and pattern of hair in the image. If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.",
    'What is the texture of depicted hairstyle? If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'Does the depicted hairstyle is straight or wavy or curly or kinky? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Can you describe the overall flow and directionality of strands? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Could you describe the bang of depicted hairstyle including its directionality and texture? If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'Describe the main geometric features of the hairstyle depicted in the image. If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'Is the length of hairstyle buzz cut, pixie, ear length, chin length, neck length, shoulder length, armpit length or mid-back length? If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Describe actors with similar hairstyle type. If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Does the haistyle cover any parts of the face? Write which exactly parts. If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'In what ways is this hairstyle a blend or combination of other popular hairstyles? If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Could you provide the most closest types of hairstyles from which this one could be blended? If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'How adaptable is this hairstyle for various occasions (casual, formal, athletic)? If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'How is this hairstyle perceived in different social or professional settings?If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image.After answer please clear you history.',
    'Are there historical figures who were iconic for wearing this hairstyle?If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
    'Could you describe the partition of this hairstyle if it is visible?If you are not sure say it honestly. Do not imagine any contents that is no in the image.If you are not sure say it honestly. Do not imagine any contents that is no in the image. After answer please clear you history.',
]


def obtain_blip_features(hairstyle_desc, model_feature_extractor,  txt_processors):
    sample = {"text_input": [txt_processors["eval"](hairstyle_desc)]}
    features_text = model_feature_extractor.extract_features(sample, mode="text")
    
    return features_text.text_embeds[0]


def obtain_description_embedding(hairstyle_description, average_descriptions, model_feature_extractor, txt_processors):
    if average_descriptions:
        dict_lists_frontal = ['From frontal view image depicts ' + hairstyle_description]
        dict_lists_back = ['From back view image depicts ' + hairstyle_description]
        
        # obtain text embedding for descriptions
        emb_frontal_first = obtain_blip_features(dict_lists_frontal[0], model_feature_extractor, txt_processors)
        emb_back_first = obtain_blip_features(dict_lists_back[0], model_feature_extractor, txt_processors)
        
        return torch.concat((emb_frontal_first, emb_back_first), 0).cuda().mean(0, keepdim=True)[None]
    
    else:
        return obtain_blip_features(hairstyle_description, model_feature_extractor, txt_processors).cuda().mean(0, keepdim=True)[None]
   