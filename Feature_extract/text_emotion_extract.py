# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
from transformers import BertTokenizer, BertModel
from pytorch_pretrained_bert import BertForMaskedLM
import pandas as pd
import pickle
from sklearn.metrics import classification_report

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert.eval()

fileName = './data/Emotion_word/English_EmotionWordsEmotionTag_Plutchik_Turner_selected_final.txt'

emotion_word = []
with open(fileName) as fr:
    for line in fr.readlines():
        emotion_list = line.split('	')
        if emotion_list[0] not in emotion_word:
            emotion_word.append(emotion_list[0])
choices=[emotion_word]
choices_idx=[]
for choice in choices:
    choice_idx=tokenizer.convert_tokens_to_ids(choice)
    choices_idx.append(choice_idx)
def pred_emotion(sentence):

    tokenized_text = tokenizer.tokenize(sentence)
    broke_point=tokenized_text.index('[MASK]')
    segments_ids=[0]*(broke_point+1)+[1]*(len(tokenized_text)-broke_point-1)
    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    segments_tensors = torch.tensor([segments_ids])

    mask_num=tokenized_text.count('[MASK]')
    mask_idxs=[idx for idx in range(len(tokenized_text)) if tokenized_text[idx]=='[MASK]']


    ans_prob=[]
    for i in range(len(choices)):
        prob = []
        for j in range(len(choices[0])):
            prob.append(0.0)
        ans_prob.append(prob)


    result = bert(ids,segments_tensors)
    for i in range(mask_num):
        mask_idx=mask_idxs[i]
        this_ans_prob = [result[0][mask_idx][choice_idx] for choice_idx in choices_idx[0]]
        ans_prob[0]=[ans_prob[0][j]+this_ans_prob[j] for j in range(len(choices[0]))]

    for i in range(len(choices)):
        for j in range(len(choices[0])):
            ans_prob[i][j]/=10

    ans_pred=[]
    for per_que in ans_prob:
        for j in range(len(per_que)):
            per_que[j] = per_que[j].cpu().detach().numpy().tolist()
            ans_pred.append(per_que[j])
    ans_pred = np.array(ans_pred)

    return ans_pred



def text_emotion(extract_dataset):

    if extract_dataset == 'Msd':
        data_file = ['../data/Msd_data/text/train.txt','./data/Msd_data/text/valid.txt','./data/Msd_data/text/test.txt',]
        all_sentence = []
        id_list = []
        for index in range(len(data_file)):
            file_path = data_file[index]
            with open(file_path) as f:
                for line in f.readlines():
                    lineLS = eval(line)
                    sentence = str(lineLS[1]).replace('#', '')
                    all_sentence.append(sentence)
                    id_list.append(lineLS[0])
    else:
        data_path = '../data/MUSTARD_data/text/MUSTARD_text.xlsx'
        data_df = pd.read_excel(data_path)
        keys = data_df['KEY']
        sentences = data_df['SENTENCE']
        i = 0
        all_sentence = []
        id_list = []
        for onetext in keys:
            if pd.notnull(onetext):
                sentence = sentences[i]
                all_sentence.append(sentence)
                id = keys[i]
                id_list.append(id)
            i += 1


    features = {}
    for i in range(len(all_sentence)):
        sentence = all_sentence[i]
        sentence = '[CLS] '+ sentence + ' Which emotion does it express? '+ '[MASK]'
        result = pred_emotion(sentence)
        features[id_list[i]] = result
    if extract_dataset == 'Msd':
        with open('feature_pkl/Msd/Msd_text_emotion_feature.pkl', 'wb') as f:
            pickle.dump(features, f)
    else:
        with open('feature_pkl/MUSTARD/MUSTARD_text_emotion_feature.pkl', 'wb') as f:
            pickle.dump(features, f)