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
torch.set_grad_enabled(False)

def visual_feature(extract_dataset):
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

    images = torch.stack([image_transforms(image) for image in images])

    ((_, image_features, _), _), _ = aclp(image=images)

    if extract_dataset == 'Msd':
        w=int(224)
        h=int(224)
        patches_list = []
        for image in images:
            img = image_transforms(image)
            patch = []
            for row in range(7):
                for col in range(7):
                    temp_patch = img[:,row*w:row*w+w,col*h:col*h+h]
                    temp_patch = torch.unsqueeze(temp_patch, dim=0)
                    patch.append(temp_patch)
            patches = torch.cat(patch, dim=0)
        patches_list.append(patches)
        patches_list = torch.cat(patches_list, dim=0)
        ((_, image_features, _,_), _), _ = aclp(image=patches_list)
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        #print('image_features:', image_features.shape)

        for k in range(len(paths_to_images)):
            ID = paths_to_images[k].split('.')[0].split('/')[1]
            one_feature = image_features[k*49:(k+1)*49, :]
            audioclip_visual_feature[ID] = np.array(one_feature)

        with open('feature_pkl/Msd/Msd_visual_feature.pkl', 'wb') as f:
            pickle.dump(audioclip_visual_feature, f)
    else:
        image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
        audioclip_visual_feature = {}
        for i in range(len(paths_to_images)):
            ID = paths_to_images[i].split('.')[0].split('/')[1]
            audioclip_visual_feature[ID] = np.array(image_features[i])
        with open('feature_pkl/MUSTARD/MUSTARD_visual_feature.pkl', 'wb') as f:
            pickle.dump(audioclip_visual_feature, f)