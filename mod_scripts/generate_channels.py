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
import functions.common_functions as common_functions
import pyximport; pyximport.install()
import mod
import time

png_folder = "/Data2TB/correctly_registered/S12/test/depth/"
out_folder = "/home/wassimea/Desktop/testing/"

data = {}
def process_image(filename, jsongt, jsonprop):
    #png_image = cv2.imread(filename, -1)
    #counter = 0

    filename_only_png = os.path.basename(filename)
    filename_only_jpg = filename_only_png.replace("png","jpg")

    print(filename_only_png)

    gtarray = common_functions.get_gt_annotations(filename_only_jpg,jsongt)
    proparray = common_functions.get_proposals(filename_only_png,jsonprop)


    #for gt_object in gtarray:
    if 1 == 1:
        gt_object = gtarray[0]
        id = gt_object[6]
        if common_functions.check_box_boundaries(gt_object[0],gt_object[1],gt_object[2],gt_object[3]):
            for prop_object in proparray:
                iou = core_functions.bb_intersection_over_union(gt_object, prop_object)
                if(iou > 0.5):
                    contained = True
                    image,head_new,right_shoulder_new,left_shoulder_new,headpoint_new = core_functions.get_new_roi_with_annotations(filename, prop_object, gt_object)
                    num_zeros = (image == 0).sum()
                    width,height = image.shape[1], image.shape[0]
                    if (width > 2 and height > 2 and num_zeros < (width*height)/ 3):
                        data['frames'].append({
                            'file' : filename_only_png.replace(".png", "_id_") + str(id) + ".jpg",
                            'width' : width,
                            'height' : height,
                            'annotations' : 
                            [
                                {
                                    'label': 'Head',
                                    'x' : head_new[0],
                                    'y' : head_new[1],
                                    'width' : head_new[2] - head_new[0],
                                    'height' : head_new[3] - head_new[1]
                                },
                                {
                                    'label' : 'Headpoint',
                                    'x' : headpoint_new[0],
                                    'y' : headpoint_new[1]
                                },
                                {
                                    'label' : 'Right Shoulder',
                                    'x' : right_shoulder_new[1],
                                    'y' : right_shoulder_new[0]
                                },
                                {
                                    'label' : 'Left Shoulder',
                                    'x' : left_shoulder_new[1],
                                    'y' : left_shoulder_new[0]
                                }
                            ]

                        })
                        #startpy = time.clock()
                        #mod = core_functions.get_channels(image, headpoint_new)
                        #timepy = time.clock() - startpy
                        startcy = time.clock()
                        #image = np.ascontiguousarray(image)
                        mod1 = mod.get_channels(image, headpoint_new)
                        #cv2.imshow("x",mod1)
                        #cv2.waitKey()
                        timecy = time.clock() - startcy
                        #cv2.rectangle(mod,(head_new[0],head_new[1]),(head_new[2] ,head_new[3]),(255,0,0),3)
                        #cv2.circle(mod,(right_shoulder_new[0], right_shoulder_new[1]), 5, (0,255,0), -1)
                        #cv2.circle(mod,(left_shoulder_new[0], left_shoulder_new[1]), 5, (0,255,0), -1)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c1.png",c1)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c2.jpg",c2)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c3.jpg",c3)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/mod.jpg",mod)
                        #cv2.imwrite("mod.jpg", mod)
                        cv2.imwrite("/home/wassimea/Desktop/chl_work/mod_scripts/modtest.jpg", mod1)
                        cv2.imshow("ayre",mod1)
                        cv2.waitKey()
                        print(timecy)
                    break

def main(_):
    data['frames'] = []
    png_files = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
    with open("/Data2TB/sample/annotations.json") as f:
        jsongt = json.load(f)
    with open("/Data2TB/correctly_registered/S12/test/output.json") as f:
      jsonprop = json.load(f)
      
    #for i in range(0,len(png_files)):
    process_image(png_folder + png_files[1], jsongt, jsonprop)
    with open('/home/wassimea/Desktop/testing/test_pos_mod.json', 'w') as outfile:  
        json.dump(data, outfile, indent=4)

if __name__ == '__main__':
  tf.app.run()
