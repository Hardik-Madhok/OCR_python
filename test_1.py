# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:42:45 2021

@author: Hardik
"""
import numpy as np
from sklearn import preprocessing
from keras.models import load_model
from utils import read_images, extract_labels

'''to test the model without any biasing I have taken one image from each class out before training.
so, it can be said that these images are new for the model. Therefore, results can be seen as fair results'''
test_images, test_names = read_images('test_dataset',128,100)

test_images = np.array(test_images)
test_images = test_images.reshape(-1, 128, 100, 3)

test_labels = extract_labels(test_names)

le = preprocessing.LabelEncoder()
le.classes_ = np.load('classes.npy')

test_enc = le.transform(test_labels)

model = load_model('model.h5')   
ynew = model.predict_classes(test_images)   