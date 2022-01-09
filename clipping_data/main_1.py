# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:18:28 2021

@author: Hardik
"""
import cv2

img_1 = cv2.imread('rho_2.jpg')

size1 = img_1.shape


height = round(size1[0]/10)
width = round(size1[1]/5)

for j in range(5):
    for i in range(10):
        img2 = img_1[height*i:height*(i+1), j*width:(j+1)*width,:]
        cv2.imwrite('Dataset/rho'+str(j+5)+ str(i)+'.png',img2)