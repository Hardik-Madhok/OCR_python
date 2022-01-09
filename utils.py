# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:59:15 2022

@author: Hardik
"""
import os
import cv2 as cv

def read_images(folder_dir,w,h):
    images = []
    names = []
    for filename in os.listdir(folder_dir):
        img = cv.imread(os.path.join(folder_dir,filename))
        img = cv.resize(img, (w,h))
        if img is not None:
            names.append(filename)
            images.append(img)
    return images, names

def extract_labels(names):
    labels = []    
    for i in range(len(names)):
        filename, extension = os.path.splitext(names[i])
        label = ""
        for i in filename:
            if i.isalpha():
                label = "".join([label, i])
        labels.append(label)
    return labels