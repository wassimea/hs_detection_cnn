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
def process_image(filename, jsongt, jsonprop):
    png_image = cv2.imread(png_folder + filename, -1)
    gtarray = []
    proparray = []
    print(filename)
    counter = 0
    filename_jpg = filename.replace("png","jpg")
    if filename_jpg in jsongt and len(jsongt[filename_jpg]["annotations"]) > 0:
        if not os.path.exists(out_folder + filename.replace(".png","")):
            os.mkdir(out_folder + filename.replace(".png",""))
        for object_gt in jsongt[filename_jpg]["annotations"]:
            if object_gt["category"] == "Head":
                xmingt = object_gt["x"] #+ 5
                ymingt = object_gt["y"]
                width = object_gt["width"]
                height = object_gt["height"]
                xmaxgt = xmingt + width
                ymaxgt = ymingt + height
                id = object_gt["id"]
                gtarray.append([xmingt,ymingt, xmaxgt,ymaxgt,id])
        for im_object in jsonprop["proposals"]:
            if im_object["file"] == filename:
                for object_prop in im_object["objects"]:
                    x = object_prop["x"]
                    y = object_prop["y"]
                    z = object_prop["z"]
                    xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
                    proparray.append([xmiw,ymiw,xmaw,ymaw,y,x])
                #head = [object["x"],object["y"],object["x"] + object["width"],object["y"] + object["height"]]
                #right_shoulder = [-1,-1]
                #left_shoulder = [-1,-1]
                #id = object["id"]
                if not os.path.exists(out_folder + filename.replace(".png","") + "/" + str(id)):
                    os.mkdir(out_folder + filename.replace(".png","") + "/" + str(id))
                #for object_candidate in jsongt[filename_jpg]["annotations"]:
                    #if object_candidate["id"] == id and object_candidate["category"] == "Right Shoulder":
                        #right_shoulder = [object_candidate["x"],object_candidate["y"]]
                    #elif object_candidate["id"] == id and object_candidate["category"] == "Left Shoulder":
                        #left_shoulder = [object_candidate["x"],object_candidate["y"]]
        for gt_object in gtarray:
            id = gt_object[4]
            for prop_object in proparray :
                iou = bb_intersection_over_union(gt_object, prop_object)
                if(iou > 0.4):
                    contained = True
                    right_shoulder = [-1,-1]
                    left_shoulder = [-1,-1]
                    for object_candidate in jsongt[filename_jpg]["annotations"]:
                        if object_candidate["id"] == id and object_candidate["category"] == "Right Shoulder":
                            right_shoulder = [object_candidate["x"],object_candidate["y"]]
                        elif object_candidate["id"] == id and object_candidate["category"] == "Left Shoulder":
                            left_shoulder = [object_candidate["x"],object_candidate["y"]]

                    headpoint = [prop_object[4],prop_object[5]]
                    image,head_new,right_shoulder_new,left_shoulder_new,headpoint_new = get_new_roi_with_annotations(filename, prop_object, gt_object, right_shoulder,left_shoulder, headpoint)
                    num_zeros = (image == 0).sum()
                    width,height = image.shape[1], image.shape[0]
                    if (width > 2 and height > 2 and num_zeros < (width*height)/ 3):
                        data['frames'].append({
                            'file' : filename.replace(".png", "_id_") + str(id) + ".jpg",
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
                        c1,c2,c3,mod = core_functions.get_channels(image, headpoint_new)
                        #cv2.rectangle(mod,(head_new[0],head_new[1]),(head_new[2] ,head_new[3]),(255,0,0),3)
                        #cv2.circle(mod,(right_shoulder_new[0], right_shoulder_new[1]), 5, (0,255,0), -1)
                        #cv2.circle(mod,(left_shoulder_new[0], left_shoulder_new[1]), 5, (0,255,0), -1)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c1.png",c1)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c2.jpg",c2)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/c3.jpg",c3)
                        #cv2.imwrite(out_folder + filename.replace(".png","") + "/" + str(id) + "/mod.jpg",mod)
                        cv2.imwrite(out_folder.replace("testing","testing_2/") + filename.replace(".png", "_id_") + str(id) + ".jpg", mod)
                    break
                else:
                    headpoint = [prop_object[4],prop_object[5]]
                    image, headpoint_new= core_functions.get_neg_roi(filename, prop_object, headpoint)
                    width,height = image.shape[1], image.shape[0]
                    data['frames'].append({
                            'file' : filename.replace(".png", "_id_") + str(id) + ".jpg",
                            'width' : width,
                            'height' : height,
                            'annotations' : 
                            [
                            ]

                        })
                    if(width > 5 and height > 5):
                        c1,c2,c3,mod = core_functions.get_channels(image, headpoint_new)
                        cv2.imwrite(out_folder.replace("testing","testing_2/") + filename.replace(".png", "_id_") + str(proparray.index(prop_object)) + ".jpg", mod)

def main(_):  
    data['frames'] = []
    png_files = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
    with open("/Data2TB/sample/annotations.json") as f:
        jsongt = json.load(f)
    with open("/home/wassimea/Desktop/wzebb.json") as f:
      jsonprop = json.load(f)
      
    for i in range(0,len(png_files)):
        process_image(png_files[i], jsongt, jsonprop)
    with open('/home/wassimea/Desktop/chl_work/testjson.json', 'w') as outfile:  
        json.dump(data, outfile, indent=4)

if __name__ == '__main__':
  tf.app.run()
