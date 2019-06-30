# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:21:23 2019

@author: Sarthak
"""

import numpy as np
import os
import time
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pandas as pd
import random
from scipy.io import wavfile
from sklearn.preprocessing import scale
import librosa.display
import librosa
import matplotlib.pyplot as plt


path_to_file =r'C:\Users\Sarthak\Desktop\ESC-50-master\cough1.wav'
data, sr = librosa.load(path_to_file, sr=44100, mono=True)
data = scale(data)

melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
log_melspec = librosa.power_to_db(melspec, ref=np.max)  
librosa.display.specshow(log_melspec, sr=sr)
names=['NotCoughing','Coughing','NotCoughing']    
    
plt.savefig('cough3' + '.png')


from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
custom_resnet_model2 = load_model('model.h5')
img_path =r"C:\Users\Sarthak\Desktop\ESC-50-master\cough1.png"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = custom_resnet_model2.predict(x)
print(names[np.argmax(preds)])
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))