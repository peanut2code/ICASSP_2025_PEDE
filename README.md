# Multi-modal Sarcasm Detection

The source code used for our paper "PEDE: Enhance Multi-modal Sarcasm Detection in Videos via Prompted Emotion Distributions"

Requirements at least one GPU is required to run the code.
Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```shell
pip3 install -r requirements.txt
```

Running the multi-modal sarcasm detection model:

if you want to use the Video-based Multi-modal Sarcasm Dataset, you can use the command:
```shell
python Sarcasm_detection_MUSTARD.py
```

When you use the commands "python Sarcasm_detection_MUSTARD.py" which will use the feature that was extracted by us, if you want to extract the feature
by yourself, you can:

1. Extract feature use the command:

   ```shell
   python Feature_extract/feature_extract.py
   ```
   which will extract the feature from two Multi-modal Sarcasm Datasets and save the resulting feature in the "feature_pkl" folder.

2. Merge the "Emb" feature and "ProEmo" feature of the Video-based Multi-modal Sarcasm Dataset:

   ```shell
   python Feature_extract/MUSTARD_feature_merge.py
   ```
   This command will generate the MUSTARD_combin_feature.pkl which was saved in the data/MUSTARD_data folder.

Due to the limitation of the uploaded file size of the submission website, we did not send the original data in a zip file, but we put the example files in each folder. In the future, we will upload the original data and the model weight trained by us to Google Drive.

Citation

Thanks for dataset from https://github.com/GussailRaat/ACL2020-SE-MUStARD-for-Multimodal-Sarcasm-Detection
