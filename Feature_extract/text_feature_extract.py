import os
import sys
import glob
import librosa.display
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from IPython.display import Audio, display
from model import AudioCLIP
from utils.transforms import ToTensor1D
import matplotlib.pyplot as plt
import pickle
import pandas as pd

torch.set_grad_enabled(False)
MODEL_FILENAME = 'pretrain_model/AudioCLIP-Full-Training.pt'
aclp = AudioCLIP(pretrained=MODEL_FILENAME)

def text_feature(extract_dataset):
    LABELS = []
    if extract_dataset == 'Msd':
        data_file = ['./data/Msd_data/text/train.txt','./data/Msd_data/text/valid.txt','./data/Msd_data/text/test.txt']
        id_list = []
        for index in range(len(data_file)):
            file_path = data_file[index]
            with open(file_path) as f:
                for line in f.readlines():
                    lineLS = eval(line)
                    sentence = str(lineLS[1]).replace('#', '')
                    LABELS.append(sentence)
                    id_list.append(lineLS[0])
    else:
        data_path = './data/MUSTARD/text/MUSTARD_text.xlsx'
        data_df = pd.read_excel(data_path)
        keys = data_df['KEY']
        sentences = data_df['SENTENCE']
        i = 0
        id_list = []
        for onetext in keys:
            if pd.notnull(onetext):
                sentence = sentences[i]
                id = keys[i]
                if len(sentence)>75:
                    sentence = sentence[0:74]
                LABELS.append(sentence)
                id_list.append(id)
            i += 1
    text = [[label] for label in LABELS]
    audioclip_text_feature = {}
    audioclip_token_feature = {}
    ((_, _, text_features, text_token_feature), _), _ = aclp(text=text)
    for i in range(len(text)):
        ID = id_list[i]
        audioclip_text_feature[ID] = np.array(text_features[i])
        audioclip_token_feature[ID] = np.array(text_token_feature[i])
    if extract_dataset == 'Msd':
        with open('feature_pkl/Msd/Msd_text_feature.pkl', 'wb') as f:
            pickle.dump(audioclip_token_feature, f)
    else:
        with open('feature_pkl/MUSTARD/MUSTARD_text_feature.pkl', 'wb') as f:
            pickle.dump(audioclip_text_feature, f)