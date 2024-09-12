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

def audio_feature():
    MODEL_FILENAME = 'pretrain_model/AudioCLIP-Full-Training.pt'
    aclp = AudioCLIP(pretrained=MODEL_FILENAME)

    # derived from ESResNeXt
    SAMPLE_RATE = 44100
    paths_to_audio = glob.glob('../data/MUSTARD_data/audio/*.wav')
    audio = list()
    time_len = []
    for path_to_audio in paths_to_audio:
        track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
        track = np.array(track)
        time_len.append(track.shape[0])
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

    audio_transforms = ToTensor1D()
    # AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
    audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
    # textual input is processed internally, so no need to transform it beforehand

    ((audio_features, _, _), _), _ = aclp(audio=audio)
    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    audioclip_audio_feature = {}
    for i in range(len(paths_to_audio)):
        ID = paths_to_audio[i].split('.')[0].split('/')[1]
        audioclip_audio_feature[ID] = audio_features[i]
    with open('feature_pkl/MUSTARD/MUSTARD_audio_feature.pkl', 'wb') as f:
        pickle.dump(audioclip_audio_feature, f)