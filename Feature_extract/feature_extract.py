# -*- coding: utf-8 -*-
import text_feature_extract
import text_emotion_extract
import visual_emotion_extract
import visual_feature_extract
import audio_emotion_extract
import audio_feature_extract
if __name__ == '__main__':
    #Text
    # extract Msd text feature
    text_feature_extract.text_feature('Msd')
    # extract MUSTARD text feature
    text_feature_extract.text_feature('MUSTARD')

    # extract Msd text emotion feature
    text_emotion_extract.text_emotion('Msd')
    # extract MUSTARD text emotion feature
    text_emotion_extract.text_emotion('MUSTARD')

    #Visual
    # extract Msd visual feature
    visual_feature_extract.visual_feature('Msd')
    # extract MUSTARD visual feature
    visual_feature_extract.visual_feature('MUSTARD')
    # extract Msd visual emotion feature
    visual_emotion_extract.visual_emotion('Msd')
    # extract MUSTARD visual emotion feature
    visual_emotion_extract.visual_emotion('MUSTARD')

    #Audio
    # extract MUSTARD audio feature
    audio_feature_extract.audio_feature()
    # extract MUSTARD audio emotion
    audio_emotion_extract.audio_emotion()