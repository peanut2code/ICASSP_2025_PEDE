import librosa
import librosa.display
import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio, display
from model import AudioCLIP
import simplejpeg
import glob
from utils.transforms import ToTensor1D
import pickle
import os
torch.set_grad_enabled(False)

def visual_emotion(extract_dataset):
    MODEL_FILENAME = 'pretrain_model/AudioCLIP-Full-Training.pt'
    # derived from CLIP
    IMAGE_SIZE = 224
    IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
    IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

    aclp = AudioCLIP(pretrained=MODEL_FILENAME)

    if extract_dataset == 'Msd':
        paths_to_images = glob.glob('../data/Msd_data/visual/*.jpg')
    else:
        paths_to_images = glob.glob('../data/MUSTARD_data/visual/*.jpg')

    fileName = 'Emotion_word/English_EmotionWordsEmotionTag_Plutchik_Turner_selected_final.txt'
    emotion_word = []
    with open(fileName) as fr:
        for line in fr.readlines():
            emotion_list = line.split('	')
            template_emotion = emotion_list[0]
            if template_emotion not in emotion_word:
                emotion_word.append(template_emotion)
    LABELS = emotion_word
    image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])
    images = list()
    for path_to_image in paths_to_images:
        with open(path_to_image, 'rb') as jpg:
            image = simplejpeg.decode_jpeg(jpg.read())
            images.append(image)
    text = [[label] for label in LABELS]
    images = torch.stack([image_transforms(image) for image in images])
    if extract_dataset == 'Msd':
        patches_list = []
        for img_name in images:
            images = list()
            paths_to_images = [img_name]
            for path_to_image in paths_to_images:
                with open(path_to_image, 'rb') as jpg:
                    image = simplejpeg.decode_jpeg(jpg.read())
                    images.append(image)
            #patch_imgs = []
            for image in images:
                #print('image111:', image.shape)
                img = image_transforms(image)
                img = img.detach().cpu().numpy()
                patch = []
                #print('img:', img.shape)
                for row in range(7):
                    for col in range(7):
                        temp_patch = img[:,row*w:row*w+w,col*h:col*h+h]
                        temp_patch = temp_patch.transpose(1,2,0)
                        temp_patch = image_transforms(temp_patch)
                        temp_patch = torch.unsqueeze(temp_patch, dim=0)
                        patch.append(temp_patch)
                patches = torch.cat(patch, dim=0)
            patches_list.append(patches)
        patches_list = torch.cat(patches_list, dim=0)
        ((_, image_features, _), _), _ = aclp(image=patches_list)
    else:
        ((_, image_features, _), _), _ = aclp(image=images)
    
    ((_, _, text_features), _), _ = aclp(text=text)

    image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)
    logits_image_text = scale_image_text * image_features @ text_features.T
    print('\tFilename, Image\t\t\tTextual Label (Confidence)', end='\n\n')

    # calculate model confidence
    confidence = logits_image_text.softmax(dim=1)
    prob_list = []
    for i in range(len(LABELS)):
        prob_list.append(0)
    features = {}
    for image_idx in range(len(paths_to_images)):
        # acquire Top-3 most similar results
        conf_values, ids = confidence[image_idx].topk(len(LABELS))
        newindex = list(ids.cpu().numpy())
        new_conf_values = conf_values.cpu().numpy()
        for j in newindex:
            prob_list[j] = new_conf_values[newindex.index(j)]
        prob_list = np.array(prob_list)
        ID = paths_to_images[image_idx].split('.')[0].split('/')[1]
        features[ID] = prob_list
        
    if extract_dataset == 'Msd':
        with open('feature_pkl/Msd/Msd_visual_emotion_feature.pkl', 'wb') as f:
            pickle.dump(features, f)
    else:
        with open('feature_pkl/MUSTARD/MUSTARD_visual_emotion_feature.pkl', 'wb') as f:
            pickle.dump(features, f)