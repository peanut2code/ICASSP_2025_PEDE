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
torch.set_grad_enabled(False)

def audio_emotion():

    MODEL_FILENAME = 'pretrain_model/AudioCLIP-Full-Training.pt'
    aclp = AudioCLIP(pretrained=MODEL_FILENAME)

    # derived from ESResNeXt
    SAMPLE_RATE = 44100
    fileName = '../data/Emotion_word/English_EmotionWordsEmotionTag_Plutchik_Turner_selected_final.txt'

    emotion_word = []
    with open(fileName) as fr:
        for line in fr.readlines():
            emotion_list = line.split('	')
            template_emotion = emotion_list[0]
            if template_emotion not in emotion_word:
                emotion_word.append(template_emotion)
    LABELS = emotion_word
    paths_to_audio = glob.glob('../data/MUSTARD_data/audio/*.wav')
    audio = list()
    for path_to_audio in paths_to_audio:
        track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
        track = np.array(track)
        # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
        # thus, the actual time-frequency representation will be visualized
        spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
        spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
        pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
        appendlist = []
        for i in range(track.shape[0], 1014632):
            appendlist.append(0)
        appendlist = np.array(appendlist)
        track = np.concatenate((track, appendlist),axis=0)
        audio.append((track, pow_spec))

    for idx, path in enumerate(paths_to_audio):
        display(Audio(audio[idx][0], rate=SAMPLE_RATE, embed=True))

    audio_transforms = ToTensor1D()
    # AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
    audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
    # textual input is processed internally, so no need to transform it beforehand
    text = [[label] for label in LABELS]

    ((audio_features, _, _), _), _ = aclp(audio=audio)
    ((_, _, text_features), _), _ = aclp(text=text)

    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)

    logits_audio_text = scale_audio_text * audio_features @ text_features.T

    # calculate model confidence
    confidence = logits_audio_text.softmax(dim=1)
    prob_list = []
    for i in range(len(LABELS)):
        prob_list.append(0)
    features = {}
    for audio_idx in range(len(paths_to_audio)):
        # acquire Top-3 most similar results
        conf_values, ids = confidence[audio_idx].topk(len(LABELS))
        newindex = list(ids.cpu().numpy())
        new_conf_values = conf_values.cpu().numpy()
        for j in newindex:
            prob_list[j] = new_conf_values[newindex.index(j)]
        prob_list = np.array(prob_list)
        ID = paths_to_audio[audio_idx].split('.')[0].split('/')[1]
        features[ID] = prob_list

    with open('feature_pkl/MUSTARD/MUSTARD_audio_emotion_feature.pkl', 'wb') as f:
        pickle.dump(features, f)