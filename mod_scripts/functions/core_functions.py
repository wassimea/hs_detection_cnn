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

#@cython.boundscheck(False)

camera_factor = 1
camera_cx = 325.5
camera_cy = 253.5
camera_fx = 518.0   
camera_fy = 519.0


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea)# + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def get_new_roi_with_annotations(filename,prop_object, gt_object):
    image = cv2.imread(filename, -1)

    head = [gt_object[0],gt_object[1],gt_object[2],gt_object[3]]
    headpoint = [prop_object[4],prop_object[5]]
    right_shoulder = gt_object[4]
    left_shoulder = gt_object[5]

    xmin, ymin, xmax, ymax = [0,0,0,0]
    head_new = [0,0,0,0]
    right_shoulder_new = [0,0]
    left_shoulder_new = [0,0]
    headpoint_new = [0,0]
    if right_shoulder[0] < prop_object[0] or left_shoulder[0] < prop_object[0] or gt_object[0] < prop_object[0]:
        xmin = min([right_shoulder[0],left_shoulder[0]], gt_object[0])
    else:
        xmin = prop_object[0]
    if right_shoulder[0] > prop_object[2] or left_shoulder[0] > prop_object[2] or gt_object[2] > prop_object[2]:
        xmax = max([right_shoulder[0],left_shoulder[0]],gt_object[2])
    else:
        xmax = prop_object[2]

    #if right_shoulder[0] != -1 and right_shoulder[1] != -1:
    xmin = xmin
    xmax = xmax
    ymin = prop_object[1]
    ymax = max([right_shoulder[1],left_shoulder[1]])

    width = xmax - xmin
    height = ymax - ymin
    while (height != width):
        if(width > height):
            ymax = ymax + 1
            height = ymax - ymin
        elif(height > width):
            xmax = xmax + 1
            width = xmax - xmin
    head_new[0] = head[0] - xmin
    head_new[1] = head[1] - ymin
    head_new[2] = head[2] - xmin
    head_new[3] = head[3] - ymin

    right_shoulder_new[0] = right_shoulder[0] - xmin
    right_shoulder_new[1] = right_shoulder[1] - ymin

    left_shoulder_new[0] = left_shoulder[0] - xmin
    left_shoulder_new[1] = left_shoulder[1] - ymin
    headpoint_new[0] = headpoint[0] - ymin
    headpoint_new[1] = headpoint[1] - xmin
    roi = image[ymin:ymax, xmin:xmax]
    return roi,head_new,right_shoulder_new,left_shoulder_new, headpoint_new

def get_channels(image, headpoint):
    zabri = image[:,10]
    x = zabri.data
    width,height = image.shape[1], image.shape[0]
    c2 = np.zeros((height,width,1), np.uint16)
    c3 = np.zeros((height,width,1), np.uint16)

    d1 = np.full((height,width,1),-1, np.uint16)
    d2 = np.full((height,width,1),-1, np.uint16)

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
    for j in range (1,width - 1):
        for i in range(1,height - 1):
            d1x = 0.5 * (image[i + 1,j] - image[i - 1, j])
            d1y = 0.5 * (image[i, j + 1] - image[i, j - 1])
            d1[i,j] = np.sqrt((d1x*d1x) + (d1y*d1y))

            d2xx = image[i + 1,j] - 2 * image[i,j] + image[i - 1, j]
            d2yy = image[i, j + 1] - 2 * image[i,j] + image[i, j - 1]
            d2xy = image[i + 1, j + 1] + image[i,j] - image[i,j + 1] - image[i + 1, j]
            if d2xx == d2yy:
                x = 1
            theta = 0.5 * np.arctan((2 * d2xy) / (d2xx - d2yy))
            d2[i,j] = (d2xx * (np.cos(theta) * np.cos(theta))) + (2 * d2xy * np.cos(theta) * np.sin(theta)) + (d2yy * (np.sin(theta) * np.sin(theta)))
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
            c2[i,j] = int(points * lightness)
            c3[i,j] = GetDist(image, headpoint,[i,j] )
    mod = cv2.merge([image, c2 ,c3])
    #cv2.normalize(mod,  mod, 0, 255, cv2.NORM_MINMAX)
    return mod

def process_d1_d2(d1,d2,i,j,image):
    d1x = 0.5 * (image[i + 1,j] - image[i - 1, j])
    d1y = 0.5 * (image[i, j + 1] - image[i, j - 1])
    d1[i,j] = np.sqrt((d1x*d1x) + (d1y*d1y))

    d2xx = image[i + 1,j] - 2 * image[i,j] + image[i - 1, j]
    d2yy = image[i, j + 1] - 2 * image[i,j] + image[i, j - 1]
    d2xy = image[i + 1, j + 1] + image[i,j] - image[i,j + 1] - image[i + 1, j]
    if d2xx == d2yy:
        x = 1
    theta = 0.5 * np.arctan((2 * d2xy) / (d2xx - d2yy))
    d2[i,j] = (d2xx * (np.cos(theta) * np.cos(theta))) + (2 * d2xy * np.cos(theta) * np.sin(theta)) + (d2yy * (np.sin(theta) * np.sin(theta)))
    d1[i,j] = np.sqrt((d1x*d1x) + (d1y*d1y))

def GetDist(img, p1, p2):
    x1 = 0
    y1 = 0
    z1 = 0
    x2 = 0
    y2 = 0
    z2 = 0


    d1 = img[p1[0],p1[1]]
    d2 = img[p2[0],p2[1]]
    if (d1 > 800 and d1 < 10000):
        z1 = float(d1) / camera_factor
        #x1 = (p1[0] - camera_cx) * z1 / camera_fx
        y1 = (p1[0] - camera_cy) * z1 / camera_fy
        #print(y1)
    if (d2 > 800 and d2 < 10000):
        z2 = float(d2) / camera_factor
        #x2 = (p2[0] - camera_cx) * z2 / camera_fx
        y2 = (p2[0] - camera_cy) * z2 / camera_fy
        #print(y2)
    dist = 0
    #dist = np.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2) *(z1 - z2))
    dist = np.sqrt((y1 - y2)*(y1 - y2))# + (z1 - z2) *(z1 - z2))
    #dist = dist/255
    #print(dist)
    return dist

def get_bounding_box_WASSIMEA(x,y,z):
  factor = 1600 / z 
  xmin = int(x - (120 * factor / 2))
  ymin = int(y - (120 * (factor / 4)))
  xmax = int(xmin + (150 * factor))
  ymax = int(ymin + (150 * factor))
  return xmin, ymin, xmax, ymax

def get_neg_roi(filename,object_prop, headpoint):
    headpoint_new = [0,0]
    xmin, ymin, xmax, ymax = [object_prop[0],object_prop[1],object_prop[2],object_prop[3]]
    headpoint_new[0] = headpoint[0] - ymin
    headpoint_new[1] = headpoint[1] - xmin
    image = cv2.imread(filename, -1)
    roi = image[ymin:ymax, xmin:xmax]
    return roi, headpoint_new

def GetSizeInImageBySizeIn3D(iSizeIn3D, iDistance):
	if (iDistance == 0 or iSizeIn3D == 0):
		return 0
	dConstFactor = 0.0 # constant, representing a line with ? length in the space, which the distance between the line and camera is 1 mm\
	bIsFactorComputed = False # unuseful 因为只想在第一次计算一次常数，以后不再计算，本变量保存计算状态
	if not (bIsFactorComputed):
		d3DDistance = 173.66668978426750 #;//std::sqrt((long double)((rX1 - rX2) * (rX1 - rX2) + (rY1 - rY2) * (rY1 - rY2) + (rZ1 - rZ2) * (rZ1 - rZ2)));
		dConstFactor = d3DDistance / (1000 * 100)
		bIsFactorComputed = True
	return int((float(iSizeIn3D) / float(iDistance) / dConstFactor))