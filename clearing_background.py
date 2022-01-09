# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:22:45 2021

@author: Hardik
"""
import numpy as np
import os 
import cv2 as cv
from utils import read_images, extract_labels


original_images, names = read_images('main_dataset', 500,400)
labels = extract_labels(names)
out_fold = 'E:/Upwork/Handwritten_numbers/Symbols/new_dataset_1'

for m in range(len(original_images)):
    img_gray = cv.cvtColor(original_images[m], cv.COLOR_BGR2GRAY)  #
    img_gray_1 = img_gray[:]
    '''checking borders and comparing it to the mean value of the image'''
    corner_1 = np.mean(img_gray[0:3,0:3])
    corner_2 = np.mean(img_gray[len(img_gray)-4:len(img_gray), 0:3])
    corner_3 = np.mean(img_gray[0:3,len(img_gray[0])-4:len(img_gray[0])])
    corner_4 = np.mean(img_gray[len(img_gray)-4:len(img_gray),len(img_gray[0])-4:len(img_gray[0])])
    gray_mean = np.mean(img_gray)
    '''if noise is present in the corners then the loop below will replace it with the mean value which is generally white of paper
    here is noise is mostly the table behind the paper'''
    if abs(corner_1-gray_mean)>30 or abs(corner_2-gray_mean)>30 or abs(corner_3-gray_mean)>30 or abs(corner_4-gray_mean)>30:
    
        img_blur = cv.GaussianBlur(img_gray, (31,31), 0)
        
        min_value = np.amin(img_blur)
        _, img_bw = cv.threshold(img_blur, min_value+30, 255, cv.THRESH_BINARY_INV)
        
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img_bw)
        
        kernel_size = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        
        
        img_bw_1 = img_bw[:]
        for i in range(1,nb_components):
            r,c = np.where(output==i)
            if min(r)==0 or min(c)==0 or max(r)==len(img_bw)-1 or max(c)==len(img_bw[0])-1:
                dummy_image = np.uint8(np.zeros_like(img_gray_1))
                dummy_image[np.where(output==i)]=255
                dummy_image = cv.dilate(dummy_image, kernel_size, iterations=5)
                img_gray_1[np.where(dummy_image==255)]=np.amax(img_blur)
           
    new_name = names[m]
    cv.imwrite(os.path.join(out_fold,new_name), img_gray_1)   
   