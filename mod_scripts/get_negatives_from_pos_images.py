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

png_folder = "/Data2TB/correctly_registered/S12/test/depth/"
out_folder = "/Data2TB/correctly_registered/S12/test/proposals/mod/neg/"

def process_image(filename, jsongt, jsonprop):
    png_image = cv2.imread(png_folder + filename, -1)

    filename_only_png = os.path.basename(filename)
    filename_only_jpg = filename_only_png.replace("png","jpg")

    print(filename_only_png)

    gtarray = common_functions.get_gt_annotations(filename_only_jpg,jsongt)
    proparray = common_functions.get_proposals(filename_only_png,jsonprop)

                #head = [object["x"],object["y"],object["x"] + object["width"],object["y"] + object["height"]]
                #right_shoulder = [-1,-1]
                #left_shoulder = [-1,-1]
                #id = object["id"]
                #if not os.path.exists(out_folder + filename.replace(".png","") + "/" + str(id)):
                #    os.mkdir(out_folder + filename.replace(".png","") + "/" + str(id))
                #for object_candidate in jsongt[filename_jpg]["annotations"]:
                    #if object_candidate["id"] == id and object_candidate["category"] == "Right Shoulder":
                        #right_shoulder = [object_candidate["x"],object_candidate["y"]]
                    #elif object_candidate["id"] == id and object_candidate["category"] == "Left Shoulder":
                        #left_shoulder = [object_candidate["x"],object_candidate["y"]]
    count = 0
    for prop_object in proparray :
        contained = False
        for gt_object in gtarray :
            iou = core_functions.bb_intersection_over_union(gt_object, prop_object)
            if(iou > 0.3):
                contained = True
        if contained == False:
            headpoint = [prop_object[4],prop_object[5]]
            image, headpoint_new = core_functions.get_neg_roi(filename, prop_object, headpoint)
            num_zeros = (image == 0).sum()
            width,height = image.shape[1], image.shape[0]
            if (width > 30 and height > 30 and num_zeros < (width*height)/ 3):
                c1,c2,c3,mod = core_functions.get_channels(image, headpoint_new)
                #cv2.rectangle(mod,(head_new[0],head_new[1]),(head_new[2] ,head_new[3]),(255,0,0),3)
                #cv2.circle(mod,(right_shoulder_new[0], right_shoulder_new[1]), 5, (0,255,0), -1)
                #cv2.circle(mod,(left_shoulder_new[0], left_shoulder_new[1]), 5, (0,255,0), -1)
                #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c1.png",c1)
                #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c2.jpg",c2)
                #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c3.jpg",c3)
                #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/mod.jpg",mod)
                cv2.imwrite(out_folder + filename_only_png.replace(".png", "_id_") + str(count) + ".jpg", mod)
                count = count + 1

def main(_):  
    png_files = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
    with open("/Data2TB/sample/annotations.json") as f:
        jsongt = json.load(f)
    with open("/Data2TB/correctly_registered/S12/test/output.json") as f:
      jsonprop = json.load(f)
    for i in range(0,len(png_files)):
        process_image(png_folder + png_files[i], jsongt, jsonprop)

if __name__ == '__main__':
  tf.app.run()
