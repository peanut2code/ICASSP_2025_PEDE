# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import random
import torch
import pickle
def pickle_loader(filename: str):
    with open(filename, "rb") as file:
        return pickle.load(file, encoding="latin1")

def make_key_list(excel_keys):
    key_list = []
    for key in excel_keys:
        if pd.notnull(key):
            if 'utterances' not in str(key):
                index = str(key).find('##')
                newkey = str(key)[0:index]
                if newkey not in key_list:
                    key_list.append(newkey)
            else:
                index = str(key).find('_utterances')
                newkey = str(key)[0:index]
                if newkey not in key_list:
                    key_list.append(newkey)
    return key_list

def make_videoids(key_list):
    print('make_videoIDS')
    videoids = {}
    for i in range(len(key_list)):
        onekeylist = []
        onekeylist.append(str(key_list[i]) + '_context')
        onekeylist.append(str(key_list[i]) + '_unterance')
        videoids[key_list[i]] = onekeylist
    return videoids

def make_speaker(key_list):
    speakers = {}
    for i in range(len(key_list)):
        onespeakerlist = []
        onespeakerlist.append(np.array([0, 1]))
        onespeakerlist.append(np.array([1, 0]))
        speakers[key_list[i]] = onespeakerlist
    return speakers

def make_sarcasms(key_list, sarcasm_labels):
    sarcasms = {}
    for i in range(len(key_list)):
        onelabel = []
        onelabel.append(0)
        for j in range(len(excel_keys)):
            if 'utterances' not in str(excel_keys[j]):
                flag = 1
            else:
                index = str(excel_keys[j]).find('_utterances')
                newkey = str(excel_keys[j])[0:index]
                if newkey == key_list[i]:
                    # print(excel_keys[j])
                    onelabel.append(int(sarcasm_labels[j]))
        sarcasms[key_list[i]] = np.array(onelabel)
    print('sarcasms:',sarcasms)
    return sarcasms

def make_implicitsLabels(key_list, excel_keys, sentiment_implicit_labels):
    implicits = {}
    for i in range(len(key_list)):
        onelabel = []
        onelabel.append(0)
        for j in range(len(excel_keys)):
            if 'utterances' in str(excel_keys[j]):
                index = str(excel_keys[j]).find('_utterances')
                newkey = str(excel_keys[j])[0:index]
                if newkey == key_list[i]:
                    if int(sentiment_implicit_labels[j]) == -1:
                        onelabel.append((int(0)))
                    if int(sentiment_implicit_labels[j]) == 0:
                        onelabel.append((int(1)))
                    if int(sentiment_implicit_labels[j]) == 1:
                        onelabel.append((int(2)))
        implicits[key_list[i]] = np.array(onelabel)
    return implicits

def make_explicitLabels(key_list, excel_keys, sentiment_explicit_labels):
    explicits = {}
    for i in range(len(key_list)):
        onelabel = []
        onelabel.append(0)
        for j in range(len(excel_keys)):
            if 'utterances' in str(excel_keys[j]):
                index = str(excel_keys[j]).find('_utterances')
                newkey = str(excel_keys[j])[0:index]
                if newkey == key_list[i]:
                    if int(sentiment_explicit_labels[j]) == -1:
                        onelabel.append((int(0)))
                    if int(sentiment_explicit_labels[j]) == 0:
                        onelabel.append((int(1)))
                    if int(sentiment_explicit_labels[j]) == 1:
                        onelabel.append((int(2)))
        explicits[key_list[i]] = np.array(onelabel)
    return explicits

def make_text_feature(excel_keys,key_list):
    text_feature = {}
    originkey = []
    for i in range(len(excel_keys)):
        if pd.notnull(excel_keys[i]):
            originkey.append(excel_keys[i])
    for i in range(len(key_list)):
        path = text_feature_path
        testdata = pickle.load(open(path, 'rb'), encoding='latin1')
        keys = testdata.keys()
        contexttensor = []
        unterancetensor = []
        for key in keys:
            if 'utterances' not in str(key):
                index = str(key).find('##')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    contexttensor.append(testdata[key])
            else:
                index = str(key).find('_utterances')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    unterancetensor.append(testdata[key])
        numtensor = contexttensor[0]
        for j in range(1, len(contexttensor)):
            numtensor = numtensor + contexttensor[j]
        ave_tensor = numtensor / len(contexttensor)
        text_feature[key_list[i]] = [ave_tensor, unterancetensor[0]]
    return text_feature

def make_text_emotion_feature(excel_keys,key_list):
    text_feature = {}
    originkey = []
    for i in range(len(excel_keys)):
        if pd.notnull(excel_keys[i]):
            originkey.append(excel_keys[i])
    for i in range(len(key_list)):
        path = text_emotion_feature_path
        testdata = pickle.load(open(path, 'rb'), encoding='latin1')
        keys = testdata.keys()
        contexttensor = []
        unterancetensor = []
        for key in keys:
            if 'utterances' not in str(key):
                index = str(key).find('##')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    contexttensor.append(testdata[key])
            else:
                index = str(key).find('_utterances')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    unterancetensor.append(testdata[key])
        numtensor = contexttensor[0]
        for j in range(1, len(contexttensor)):
            numtensor = numtensor + contexttensor[j]
        ave_tensor = numtensor / len(contexttensor)
        text_feature[key_list[i]] = [ave_tensor, unterancetensor[0]]
    return text_feature

def make_visual_feature(excel_keys,key_list):
    visual_feature = {}
    all_test_data = {}
    path = visual_feature_path
    for file_index in range(len(path)):
        testdata = pickle.load(open(path[file_index], 'rb'), encoding='latin1')
        keys = testdata.keys()
        for keyindex in keys:
            all_test_data[keyindex] = testdata[keyindex]
    for i in range(len(key_list)):
        contexttensor = []
        unterancetensor = []
        keys = all_test_data.keys()
        for key in keys:
            if 'utterance' not in str(key):
                index = str(key).find('_c')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    contexttensor.append(all_test_data[key])
            else:
                index = str(key).find('_utterance')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    unterancetensor.append(all_test_data[key])
        seqlen = 30
        now_cont_len = len(contexttensor)
        if now_cont_len < seqlen:
            for h in range(now_cont_len, seqlen):
                contexttensor.append(contexttensor[now_cont_len - 1])
        now_utter_len = len(unterancetensor)
        if now_utter_len < seqlen:
            for h in range(now_utter_len, seqlen):
                unterancetensor.append(unterancetensor[now_utter_len - 1])
        visual_feature[key_list[i]] = [[contexttensor], [unterancetensor]]
    return visual_feature

def make_visual_emotion_feature(excel_keys,key_list):
    visual_emotion_feature = {}
    all_test_data = {}
    path = visual_emotion_feature_path
    for file_index in range(len(path)):
        testdata = pickle.load(open(path[file_index], 'rb'), encoding='latin1')
        keys = testdata.keys()
        for keyindex in keys:
            all_test_data[keyindex] = testdata[keyindex]
    for i in range(len(key_list)):
        contexttensor = []
        unterancetensor = []
        keys = all_test_data.keys()
        for key in keys:
            if 'utterance' not in str(key):
                index = str(key).find('_c')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    contexttensor.append(all_test_data[key])
            else:
                index = str(key).find('_utterance')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    unterancetensor.append(all_test_data[key])
        seqlen = 30
        now_cont_len = len(contexttensor)
        if now_cont_len < seqlen:
            for h in range(now_cont_len, seqlen):
                contexttensor.append(contexttensor[now_cont_len - 1])
        now_utter_len = len(unterancetensor)
        if now_utter_len < seqlen:
            for h in range(now_utter_len, seqlen):
                unterancetensor.append(unterancetensor[now_utter_len - 1])
        visual_emotion_feature[key_list[i]] = [[contexttensor], [unterancetensor]]
    return visual_emotion_feature

def make_audio_feature(excel_keys,key_list):
    audio_feature = {}
    originkey = []
    for i in range(len(excel_keys)):
        if pd.notnull(excel_keys[i]):
            originkey.append(excel_keys[i])
    for i in range(len(key_list)):
        path = audio_feature_path
        testdata = pickle.load(open(path, 'rb'), encoding='latin1')
        keys = testdata.keys()
        contexttensor = []
        unterancetensor = []
        for key in keys:
            if 'utterances' not in str(key):
                index = str(key).find('##')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    if testdata[key] != []:
                        contexttensor.append(testdata[key])
            else:
                index = str(key).find('_utterances')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    if testdata[key] != []:
                        unterancetensor.append(testdata[key])
        if len(contexttensor) == 0:
            contexttensor.append( np.zeros((128, ), dtype = float) )
        if len(unterancetensor) == 0:
            unterancetensor.append( np.zeros((128, ), dtype = float) )
        numtensor = contexttensor[0]
        for j in range(1, len(contexttensor)):
            numtensor = numtensor + contexttensor[j]
        ave_tensor = numtensor / len(contexttensor)
        audio_feature[key_list[i]] = [ave_tensor, unterancetensor[0]]
    return audio_feature

def make_audio_emotion_feature(excel_keys,key_list):
    audio_emotion_feature = {}
    all_test_data = {}
    path = audio_emotion_feature_path
    for file_index in range(len(path)):
        testdata = pickle.load(open(path[file_index], 'rb'), encoding='latin1')
        keys = testdata.keys()
        for keyindex in keys:
            all_test_data[keyindex] = testdata[keyindex]
    for i in range(len(key_list)):
        contexttensor = []
        unterancetensor = []
        keys = all_test_data.keys()
        for key in keys:
            if '_c' in str(key):
                index = str(key).find('_c')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    contexttensor.append(all_test_data[key])
            else:
                index = str(key).find('_audio')
                newkey = str(key)[0:index]
                if newkey == key_list[i]:
                    unterancetensor.append(all_test_data[key])
        seqlen = 10
        now_cont_len = len(contexttensor)
        if now_cont_len < seqlen:
            for h in range(now_cont_len, seqlen):
                contexttensor.append(contexttensor[now_cont_len - 1])
        now_utter_len = len(unterancetensor)
        if now_utter_len < seqlen:
            for h in range(now_utter_len, seqlen):
                unterancetensor.append(unterancetensor[now_utter_len - 1])
        audio_emotion_feature[key_list[i]] = [[contexttensor], [unterancetensor]]
    return audio_emotion_feature

if __name__ == '__main__':

    data1 = pd.read_excel('../data/MUSTARD_data/text/MUSTARD_text.xlsx', header=0)
    data = data1.loc[:, ['KEY','SPEAKER', 'SENTENCE', 'SHOW', 'SARCASM','SENTIMENT_IMPLICIT','SENTIMENT_EXPLICIT','EMOTION_IMPLICIT','EMOTION_EXPLICIT']]
    excel_keys=data['KEY']
    sentiment_implicit_labels = data['SENTIMENT_IMPLICIT']
    sentiment_explicit_labels = data['SENTIMENT_EXPLICIT']
    sarcasm_labels=data['SARCASM']
    sentences=data['SENTENCE']
    shows = data['SHOW']

    text_feature_path = 'feature_pkl/MUSTARD/MUSTARD_text_feature.pkl'
    text_emotion_feature_path = 'feature_pkl/MUSTARD/MUSTARD_text_emotion_feature.pkl'
    visual_feature_path = 'feature_pkl/MUSTARD/MUSTARD_visual_feature.pkl'
    visual_emotion_feature_path = 'feature_pkl/MUSTARD/MUSTARD_visual_emotion_feature.pkl'
    audio_feature_path = 'feature_pkl/MUSTARD/MUSTARD_audio_feature.pkl'
    audio_emotion_feature_path = 'feature_pkl/MUSTARD/MUSTARD_audio_emotion_feature.pkl'

    allfeature=[]
    key_list = make_key_list(excel_keys)

    videoids = make_videoids(key_list)
    allfeature.append(videoids)

    #make_speaker
    print('make_speaker')
    speakers = make_speaker(key_list)
    allfeature.append(speakers)

    # make_videoLabels
    print('make_sarcasmsLabels')
    sarcasms = make_sarcasms(key_list, sarcasm_labels)
    allfeature.append(sarcasms)

    print('make_implicitsLabels')
    implicits = make_implicitsLabels(key_list, excel_keys, sentiment_implicit_labels)
    allfeature.append(implicits)

    print('make_explicitLabels')
    explicits = make_explicitLabels(key_list, excel_keys, sentiment_explicit_labels)
    allfeature.append(explicits)

    print('make_text_feature')
    text_feature = make_text_feature(excel_keys,key_list)
    allfeature.append(text_feature)

    print('make_text_emotion_feature')
    text_emotion_feature = make_text_emotion_feature(excel_keys, key_list)
    allfeature.append(text_emotion_feature)

    print('make_visual_feature')
    visual_feature = make_visual_feature(excel_keys, key_list)
    allfeature.append(visual_feature)

    print('make_visual_emotion_feature')
    visual_emotion_feature = make_visual_emotion_feature(excel_keys, key_list)
    allfeature.append(visual_emotion_feature)

    print('make_audio_feature')
    audio_feature = make_audio_feature(excel_keys, key_list)
    allfeature.append(audio_feature)

    print('make_audio_emotion_feature')
    audio_emotion_feature = make_audio_emotion_feature(excel_keys, key_list)
    allfeature.append(audio_emotion_feature)

    trainVid = set()
    testVid = set()
    valid = set()
    firend = []
    BBT = []
    GOLDENGIRLS = []
    SARCASMOHOLICS = []
    for i in range(len(excel_keys)):
        key = excel_keys[i]
        if 'utterances' in str(excel_keys[i]) and shows[i] == 'FRIENDS':
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            firend.append(newkey)
            testVid.add(newkey)
        if 'utterances' in str(excel_keys[i]) and shows[i] == 'GOLDENGIRLS':
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            GOLDENGIRLS.append(newkey)
        if 'utterances' in str(excel_keys[i]) and shows[i] == 'BBT':
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            BBT.append(newkey)
        if 'utterances' in str(excel_keys[i]) and shows[i] == 'SARCASMOHOLICS':
            index = str(key).find('_utterances')
            newkey = str(key)[0:index]
            SARCASMOHOLICS.append(newkey)

    seed_set = 1000
    random.seed(seed_set)
    validBBT = random.sample(BBT, 40)
    validGOLDENGIRLS = random.sample(GOLDENGIRLS, 20)
    validSARCASMOHOLICS = random.sample(SARCASMOHOLICS, 7)

    for b in BBT:
        if b in validBBT:
            valid.add(b)
        else:
            trainVid.add(b)
    for b in GOLDENGIRLS:
        if b in validGOLDENGIRLS:
            valid.add(b)
        else:
            trainVid.add(b)
    for b in SARCASMOHOLICS:
        if b in validSARCASMOHOLICS:
            valid.add(b)
        else:
            trainVid.add(b)

    allfeature.append(trainVid)
    allfeature.append(testVid)
    allfeature.append(valid)
    with open('../data/MUSTARD_data/MUSTARD_combin_feature.pkl', 'wb') as f:
        pickle.dump(allfeature, f)

