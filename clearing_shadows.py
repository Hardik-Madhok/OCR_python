# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:48:56 2021

@author: Hardik
"""
import numpy as np
import os 
import cv2 as cv
import math
from utils import read_images, extract_labels

'''Previous writen data has only paper in the background in this code we will clip the symbols so that it can be fed as Input'''
original_images, names = read_images('main_dataset',500,400)
labels = extract_labels(names)
images, names = read_images('new_dataset_1')
out_fold = 'E:/Upwork/Handwritten_numbers/Symbols/new_dataset_2'
out_fold_1 = 'E:/Upwork/Handwritten_numbers/Symbols/new_dataset_3'

for m in range(len(images)):
    img = cv.cvtColor(images[m], cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (11,11), 0)
    mean_value = np.mean(blur)
    _, img_bw = cv.threshold(blur, mean_value-25,255,cv.THRESH_BINARY_INV)
    #kernel here is taken circular as the letters are curved to this will is the best option to go with
    kernel_size = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))   
    dummy_image = np.uint8(np.zeros_like(img))
    img_bw_1 = img_bw[:]
    '''all the components that are joined with borders are removed including letters
    because that will only confuse the machine''' 
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img_bw_1)
    for i in range(1,nb_components):
        r,c = np.where(output==i)
        if min(r)==0 or min(c)==0 or max(r)==len(img_bw_1)-1 or max(c)==len(img_bw_1[0])-1:
            
            dummy_image[np.where(output==i)]=255
            dummy_image = cv.dilate(dummy_image, kernel_size, iterations=5)
    img_bw_1[np.where(dummy_image==255)]=0
    kernel = np.ones((3,3), 'uint8')
    img_bw_1 = cv.erode(img_bw_1, kernel, iterations = 2)
    img_bw_1 = cv.dilate(img_bw_1, kernel_size, iterations = 4)
    ''''it was experienced that some letters were broken after applying thresholding
    so, the distance between them were calculated so that no half latter could be clipped off'''
    nb_components_1, output_1, stats_1, centroids_1 = cv.connectedComponentsWithStats(img_bw_1)
    img_bw_2 = np.zeros_like(img_bw_1)
    if nb_components_1 > 2:
        max_area = max(stats_1[1:,4])
        ind, col = np.where(stats_1==max_area)
        valid_stats = stats_1[ind]
        img_bw_2[np.where(output_1==ind)]=255
        for i in range(1,nb_components_1):
            distance = math.dist(centroids_1[ind[0]],centroids_1[i])
            if distance<100:
                img_bw_2[np.where(output_1==i)]=255
            else:
                img_bw_2[np.where(output_1==i)]=0
                
        r,c = np.where(img_bw_2 == 255)
        min_row = min(r)
        min_col = min(c)
        max_row = max(r)
        max_col = max(c)
        
        clipped_image = original_images[m][min_row:max_row, min_col:max_col]
        clipped_image = np.uint8(clipped_image)
        clipped_image = cv.resize(clipped_image, (128,100))
                
    elif nb_components_1 ==2:
        area = stats_1[1,4]
        if area >= 350:
            img_bw_2[np.where(output_1==1)]=255
        
            r,c = np.where(img_bw_2 == 255)
            min_row = min(r)
            min_col = min(c)
            max_row = max(r)
            max_col = max(c)
            
            clipped_image = original_images[m][min_row:max_row, min_col:max_col]
            clipped_image = np.uint8(clipped_image)
            clipped_image = cv.resize(clipped_image, (128,100))
            
        else:
            clipped_image = np.zeros_like((128,100))
            clipped_image = np.uint8(clipped_image)
                
    else:
        clipped_image = np.zeros_like((128,100))
        clipped_image = np.uint8(clipped_image)

    
    new_name = names[m]
    cv.imwrite(os.path.join(out_fold_1,new_name), clipped_image)  