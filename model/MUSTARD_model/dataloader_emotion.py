# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import torch
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
seed_torch(20210412)

class MUSTARDDataset(Dataset):

    def __init__(self, path, flag='train'):
        with open(path, 'rb') as file:
            data=pickle.load(file, encoding='latin1')
        self.videoIDs = data[0]
        self.videoSpeakers = data[1]
        self.sarcasmslabel = data[2]
        self.sentiment_implicit = data[3]
        self.sentiment_explicit = data[4]
        self.text_feature = data[5]
        self.text_emotion_feature = data[6]
        self.visual_feature = data[7]
        self.visual_emotion_feature = data[8]
        self.audio_feature = data[9]
        self.audio_emotion_feature = data[10]
        self.trainVid = sorted(data[11])
        self.testVid = sorted(data[12])
        self.valid = sorted(data[13])

        if flag == 'train':
            self.keys = [x for x in self.trainVid]
        if flag == 'test':
            self.keys = [x for x in self.testVid]
        if flag == 'valid':
            self.keys = [x for x in self.valid]
        self.len = len(self.keys)

    def __getitem__(self, index):

        vid = self.keys[index]
        #true
        umask=[]
        labellen=len(self.sarcasmslabel[vid])
        for i in range (labellen):
            if i!= labellen-1:
                umask.append(0)
            else:
                umask.append(1)
        # print('self.visual_feature[vid] shape:', self.visual_feature[vid][0].shape)
        # print('len self.text_emotion_feature):', len(self.text_emotion_feature))
        # print('len self.text_emotion_feature[vid]:',  self.text_emotion_feature[vid][0].shape)
        return torch.FloatTensor(self.text_feature[vid]), \
               torch.FloatTensor(self.text_emotion_feature[vid]), \
               torch.FloatTensor(self.visual_feature[vid]), \
               torch.FloatTensor(self.visual_emotion_feature[vid]), \
               torch.FloatTensor(self.audio_feature[vid]), \
               torch.FloatTensor(self.audio_emotion_feature[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor(umask), \
               torch.LongTensor(self.sarcasmslabel[vid]), \
               torch.LongTensor(self.sentiment_implicit[vid]), \
               torch.LongTensor(self.sentiment_explicit[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<11 else dat[i].tolist() for i in dat]