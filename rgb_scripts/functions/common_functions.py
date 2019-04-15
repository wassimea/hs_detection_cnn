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
import functions.core_functions as core_functions


def get_gt_annotations(filename, jsongt):
    gtarray = []
    right_shoulder = [-1,-1]
    left_shoulder = [-1,-1]
    if filename in jsongt and len(jsongt[filename]["annotations"]) > 0:
        for object_gt in jsongt[filename]["annotations"]:
            if object_gt["category"] == "Head":
                xmingt = object_gt["x"] #+ 5
                ymingt = object_gt["y"]
                width = object_gt["width"]
                height = object_gt["height"]
                xmaxgt = xmingt + width
                ymaxgt = ymingt + height
                id = object_gt["id"]
                for shoulder_candidate in jsongt[filename]["annotations"]:
                    if shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Right Shoulder":
                        right_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                    elif shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Left Shoulder":
                        left_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                gtarray.append([xmingt,ymingt, xmaxgt,ymaxgt,right_shoulder,left_shoulder, id])
    return gtarray

def get_proposals(filename, jsonprop):
    proparray = []
    for im_object in jsonprop:
        if im_object["file"] == filename:
            for object_prop in im_object["objects"]:
                x = object_prop["x"]
                y = object_prop["y"]
                z = object_prop["z"]
                xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
                proparray.append([xmiw,ymiw,xmaw,ymaw,y,x])
    return proparray

def check_box_boundaries(xmin,ymin,xmax,ymax):
    if xmin > 48 and ymin > 30 and xmax < 550 and ymax < 415:
        return True
    else:
        return False

def check_if_box_is_contained(rect1,rect2):
    xmin1 = rect1[0]
    ymin1 = rect1[1]
    xmax1 = rect1[2]
    ymax1 = rect1[3]

    xmin2 = rect2[0]
    ymin2 = rect2[1]
    xmax2 = rect2[2]
    ymax2 = rect2[3]

    if(xmin2 <= xmin1 and ymin2 <= ymin1 and xmax2 >= xmax1 and ymax2 >= ymax1):
        return True
    else:
        return False


