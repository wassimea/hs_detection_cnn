import tensorflow as tf
import sys
import os
import json
import PIL
import numpy as np
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import multiprocessing
import cv2
from skimage.filters import rank
from skimage.morphology import rectangle
import cython

#%%cython -a
@cython.boundscheck(False)

cpdef get_channels(unsigned char[:,:] image, Point headpoint):
    cdef int width,height = image.shape[1], image.shape[0]
    c2 = np.zeros((height,width,1), np.uint16)
    c3 = np.zeros((height,width,1), np.uint16)

    d1 = np.zeros((height,width,1), np.uint16)
    d2 = np.zeros((height,width,1), np.uint16)

    mod = np.zeros((height,width,3), np.uint16)
    d1[0,0] = 0
    d1[0,width - 1] = 0
    d1[height - 1,0] = 0
    d1[height - 1,width - 1] = 0

    c2[0,0] = 0
    c2[0,width - 1] = 0
    c2[height - 1,0] = 0
    c2[height - 1,width - 1] = 0

    lightness = 255/24
    #headpoint_j = int(height/5)
    #headpoint_i = int(width/2)
    #while image[headpoint_j, headpoint_i] == 0:
        #headpoint_j = headpoint_j + 1
    #headpoint = [headpoint_j, headpoint_i]
    for j in range (1,width - 1):
        for i in range(1,height - 1):
            d1x = 0.5 * (image[i + 1,j] - image[i - 1, j])
            d1y = 0.5 * (image[i, j + 1] - image[i, j - 1])
            d1[i,j] = np.sqrt((d1x*d1x) + (d1y*d1y))

            d2xx = image[i + 1,j] - 2 * image[i,j] + image[i - 1, j]
            d2yy = image[i, j + 1] - 2 * image[i,j] + image[i, j - 1]
            d2xy = image[i + 1, j + 1] + image[i,j] - image[i,j + 1] - image[i + 1, j]
            theta = 0.5 * np.arctan((2 * d2xy) / (d2xx - d2yy))
            d2[i,j] = (d2xx * (np.cos(theta) * np.cos(theta))) + (2 * d2xy * np.cos(theta) * np.sin(theta)) + (d2yy * (np.sin(theta) * np.sin(theta)))
            if(d1x != 0 or d1y != 0):
                tollshway = 0   
            d1[i,j] = np.sqrt((d1x*d1x) + (d1y*d1y))
    for j in range (1,width - 1):
        for i in range(1,height - 1):
            points = 0

            if(image[i,j] - image[i - 1, j + 1] >3 and image[i,j] - image[i + 1, j + 1] >3):
                points = points + 1
            if(image[i,j] - image[i - 1, j - 1] >3 and image[i,j] - image[i - 1, j + 1] >3):
                points = points + 1
            if(image[i,j] - image[i - 1, j - 1] >3 and image[i,j] - image[i + 1, j - 1] >3):
                points = points + 1
            if(image[i,j] - image[i + 1, j - 1] >3 and image[i,j] - image[i + 1, j + 1] >3):
                points = points + 1
            if(image[i,j] - image[i - 1, j] >3 and image[i,j] - image[i, j + 1] >3):
                points = points + 1
            if(image[i,j] - image[i, j - 1] >3 and image[i,j] - image[i - 1, j] >3):
                points = points + 1
            if(image[i,j] - image[i, j - 1] >3 and image[i,j] - image[i + 1, j] >3):
                points = points + 1
            if(image[i,j] - image[i, j + 1] >3 and image[i,j] - image[i + 1, j] >3):
                points = points + 1

            if(d1[i,j] - d1[i - 1, j + 1] >3 and d1[i,j] - d1[i + 1, j + 1] >3):
                points = points + 1
            if(d1[i,j] - d1[i - 1, j - 1] >3 and d1[i,j] - d1[i - 1, j + 1] >3):
                points = points + 1
            if(d1[i,j] - d1[i - 1, j - 1] >3 and d1[i,j] - d1[i + 1, j - 1] >3):
                points = points + 1
            if(d1[i,j] - d1[i + 1, j - 1] >3 and d1[i,j] - d1[i + 1, j + 1] >3):
                points = points + 1
            if(d1[i,j] - d1[i - 1, j] >3 and d1[i,j] - d1[i, j + 1] >3):
                points = points + 1
            if(d1[i,j] - d1[i, j - 1] >3 and d1[i,j] - d1[i - 1, j] >3):
                points = points + 1
            if(d1[i,j] - d1[i, j - 1] >3 and d1[i,j] - d1[i + 1, j] >3):
                points = points + 1
            if(d1[i,j] - d1[i, j + 1] >3 and d1[i,j] - d1[i + 1, j] >3):
                points = points + 1

            if(d2[i,j] - d2[i - 1, j + 1] >3 and d2[i,j] - d2[i + 1, j + 1] >3):
                points = points + 1
            if(d2[i,j] - d2[i - 1, j - 1] >3 and d2[i,j] - d2[i - 1, j + 1] >3):
                points = points + 1
            if(d2[i,j] - d2[i - 1, j - 1] >3 and d2[i,j] - d2[i + 1, j - 1] >3):
                points = points + 1
            if(d2[i,j] - d2[i + 1, j - 1] >3 and d2[i,j] - d2[i + 1, j + 1] >3):
                points = points + 1
            if(d2[i,j] - d2[i - 1, j] >3 and d2[i,j] - d2[i, j + 1] >3):
                points = points + 1
            if(d2[i,j] - d2[i, j - 1] >3 and d2[i,j] - d2[i - 1, j] >3):
                points = points + 1
            if(d2[i,j] - d2[i, j - 1] >3 and d2[i,j] - d2[i + 1, j] >3):
                points = points + 1
            if(d2[i,j] - d2[i, j + 1] >3 and d2[i,j] - d2[i + 1, j] >3):
                points = points + 1

            zibbe = int(points * lightness)
            c2[i,j] = int(points * lightness)
            c3[i,j] = GetDist(image, headpoint,[i,j] )


    #image = (image/256).astype('uint8')
    kernel = np.ones((5,5),np.float32)/25
    #c2 = cv2.filter2D(c2,-1,kernel)
    #c2 = rank.mean(c2, rectangle(3,3), mask=c2!=0)
    #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c1.png",image)
    #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c2.jpg",c2)
    #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c3.jpg",c3)
    mod = cv2.merge([image, c2 ,c3])
    #mod = (mod/256).astype('uint8')
    #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/mod.jpg", mod)
    #cv2.imwrite(out_folder.replace("testing","testing_2") + filename.replace(".png", "_id_") + str(id) + ".jpg", mod)
    #counter = counter + 1
    return image,c2,c3,mod