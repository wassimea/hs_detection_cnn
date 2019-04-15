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
png_folder = "/Data2TB/correctly_registered/S12/test/depth/"
out_folder = "/home/wassimea/Desktop/tst_out/testing/"
camera_factor = 1
camera_cx = 325.5
camera_cy = 253.5
camera_fx = 518.0
camera_fy = 519.0
data = {}
def process_image():
    with open("/home/wassimea/Desktop/wzebb.json") as f:
        jsonprop = json.load(f)
    filename = "/Data2TB/correctly_registered/S12/test/depth/video_fifth_2018-04-23_CAM1_1524500433782.png"
    filename_only = "video_fifth_2018-04-23_CAM1_1524500433782.png"
    png_image = cv2.imread(filename, -1)
    for im_object in jsonprop["proposals"]:
        if im_object["file"] == filename_only:
            count = 0
            for object_prop in im_object["objects"]:
                x = object_prop["x"]
                y = object_prop["y"]
                z = object_prop["z"]
                xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
                prop_object = [xmiw,ymiw,xmaw,ymaw,y,x]
                headpoint = [y,x]
                image, headpoint_new = core_functions.get_neg_roi(filename_only, prop_object, headpoint)
                width,height = image.shape[1], image.shape[0]
                c1,c2,c3,mod = core_functions.get_channels(image, headpoint_new)
                cv2.imwrite("/home/wassimea/Desktop/tst_out/testing_2/" + filename_only.replace(".png", "_id_") + str(count) + ".jpg", mod)
                count = count + 1

def main(_):  
    with open("/home/wassimea/Desktop/wzebb.json") as f:
        jsonprop = json.load(f)
        process_image()

if __name__ == '__main__':
  tf.app.run()
