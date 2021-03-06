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

resized_proposals_folder = "/Data2TB/correctly_registered/S12/train/proposals_mod/train/"
resized_proposals_json = "/Data2TB/correctly_registered/S12/train/proposals_mod/train/train_mod.json"

def process_resized_proposals():
    with open(resized_proposals_json) as f:
        jsonprop = json.load(f)
    for frame in jsonprop["frames"]:
        filename_png = os.path.basename(frame["file"])
        filename_jpg = filename_png.replace(".png","jpg")
        rgb_image = cv2.imread(resized_proposals_folder + filename_jpg)
        print(filename_png)
        #filename = filename.replace("png","jpg")
        #filename = filename.replace("-resized-1.2-rotated-10","")
        for annotation in frame["annotations"]:
            if annotation["label"] == "Head":
                xmingt = annotation["x"]
                ymingt = annotation["y"]
                xmaxgt = xmingt + annotation["width"]
                ymaxgt = ymingt + annotation["height"]
                cv2.rectangle(rgb_image,(xmingt,ymingt),(xmaxgt ,ymaxgt),(0,0,255),3)
            elif annotation["label"] == "Headpoint":
                xhp = annotation["y"]
                yhp = annotation["x"]
                cv2.circle(rgb_image,(yhp, xhp), 5, (0,0,255), -1)
            elif annotation["label"] ==  "Right Shoulder" or annotation["label"] ==  "Left Shoulder":
                xs = annotation["x"]
                ys = annotation["y"]
                cv2.circle(rgb_image,(ys, xs), 5, (0,255,0), -1)
        cv2.imshow("ayre",rgb_image)
        cv2.waitKey()
            #cv2.imwrite(out_folder + filename,rgb_image)
    #if filename_jpg in jsongt and len(jsongt[filename_jpg]["annotations"]) > 0:
    #    for object_gt in jsongt[filename_jpg]["annotations"]:
    #        if object_gt["category"] == "Head":
    #            xmingt = object_gt["x"] #+ 5
    #            ymingt = object_gt["y"]
    #            width = object_gt["width"]
    #            height = object_gt["height"]
    #            xmaxgt = xmingt + width
    #            ymaxgt = ymingt + height
    #            id = object_gt["id"]
    #            cv2.rectangle(rgb_image,(xmingt,ymingt),(xmaxgt ,ymaxgt),(0,0,255),3)
    #            right_shoulder = [-1,-1]
    #            left_shoulder = [-1,-1]
    #            for object_candidate in jsongt[filename_jpg]["annotations"]:
    #                if object_candidate["id"] == id and object_candidate["category"] == "Right Shoulder":
    #                    right_shoulder = [object_candidate["x"],object_candidate["y"]]
    #                elif object_candidate["id"] == id and object_candidate["category"] == "Left Shoulder":
    #                    left_shoulder = [object_candidate["x"],object_candidate["y"]]
    #            cv2.circle(rgb_image,(right_shoulder[0], right_shoulder[1]), 5, (0,255,0), -1)
    #            cv2.circle(rgb_image,(left_shoulder[0], left_shoulder[1]), 5, (0,255,0), -1)
    cv2.imwrite(out_folder + filename,rgb_image)
                

def main(_):  
    process_resized_proposals()
if __name__ == '__main__':
  tf.app.run()
