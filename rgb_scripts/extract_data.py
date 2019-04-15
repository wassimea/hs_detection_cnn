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
from automation_config import config
import functions.core_functions as core_functions
import functions.common_functions as common_functions
import mod

rgb_folder = config.pos_rgb_in
pos_out_folder = config.pos_rgb_out
neg_out_folder = config.neg_rgb_out

data = {}
def process_image(filename, jsongt, jsonprop):
    #png_image = cv2.imread(filename, -1)
    #counter = 0

    filename_only_jpg = os.path.basename(filename).replace(".png",".jpg")
    filename_only_png = filename_only_jpg.replace("jpg","png")

    #print(filename_only_jpg)

    gtarray = common_functions.get_gt_annotations(filename_only_jpg,jsongt)
    proparray = common_functions.get_proposals(filename_only_png,jsonprop)

    flag_array = np.zeros(len(proparray))

    #if "videos_holland_orbbec_2018-02-21_CAM2_1519231983220" in filename:
    #    im = cv2.imread(filename)
    #    for gt in gtarray:
    #        cv2.rectangle(im,(gt[0], gt[1]), (gt[2], gt[3]),(0,255,0), 1)
    #    for prop in proparray:
    #        cv2.rectangle(im,(prop[0], prop[1]), (prop[2], prop[3]),(0,0,255), 1)
#
    #    cv2.imshow("im",im)
    #    cv2.waitKey()

    for gt_object in gtarray:
        id = gt_object[6]
        if 1 == 1:#common_functions.check_box_boundaries(gt_object[0],gt_object[1],gt_object[2],gt_object[3]):
            list_valid_proposals = []
            list_valid_proposals_ious = []
            for prop_object in proparray:
                #if(flag_array[proparray.index(prop_object)] != 1):
                iou = core_functions.bb_intersection_over_union(gt_object, prop_object)
                #if(common_functions.check_if_box_is_contained(gt_object,prop_object)):
                if(iou > 0.75):
                    list_valid_proposals.append(prop_object)
                    list_valid_proposals_ious.append(iou)
            
            if(len(list_valid_proposals_ious) > 0):
                #list_acceptable_proposals = []
                #idx = np.argsort(list_valid_proposals_ious)
#
                #list_valid_proposals_ious = np.array(list_valid_proposals_ious)[idx]
                #list_valid_proposals = np.array(list_valid_proposals)[idx]
                #list_valid_proposals_ious = list_valid_proposals_ious.tolist()
                #list_valid_proposals = list_valid_proposals.tolist()

                for i in range (0,len(list_valid_proposals)):

                    #highest_iou_index = np.argmax(list_valid_proposals_ious)
                    #prop_object = list_valid_proposals[highest_iou_index]
                    #flag_array[proparray.index(list_valid_proposals[i])] = 1

                    image,head_new,right_shoulder_new,left_shoulder_new,headpoint_new = core_functions.get_new_roi_with_annotations(filename, list_valid_proposals[i], gt_object)
                    if config.mod == True:
                        if("video_fifth_orbbec_2018-01-19_lunch_CAM1_1516381409554.png" in filename):
                            x = 1
                        #image = mod.get_channels(image,headpoint_new)#.astype('uint8')
                        image = mod.get_channels(image,headpoint_new)
                        #roi1copy = roi1.copy()
                        #b,g,r = cv2.split(image)
                        #b = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
                        #g = g.astype('uint8')
                        #r = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
                        #b = (b/256).astype('uint8')
                        #g = (g/256).astype('uint8')
                        #r = (r/256).astype('uint8')
                        #image = cv2.merge((b,g,r)).astype('uint8')
                        #cv2.imshow("im",image)
                        #cv2.waitKey()
                        #cv2.normalize(image,  image, 0, 255, cv2.NORM_MINMAX)
                        #image = image.astype('uint8')
                    if image is not None:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        result, encimg = cv2.imencode('.jpg', image, encode_param)
                        image = cv2.imdecode(encimg, 1)
                        flag_array[proparray.index(list_valid_proposals[i])] = 1
                        width,height = image.shape[1], image.shape[0]
                        if (width > 2 and height > 2):# and num_zeros < (width*height)/ 3):
                            data['frames'].append({
                                'file' : filename_only_png.replace(".png", "_id_") + str(id) + "_index_" + str(i) + ".png",
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
                            cv2.imwrite(pos_out_folder + filename_only_png.replace(".png", "_id_") + str(id) + "_index_" + str(i) + ".png", image)
    
    for i in range (0, len(flag_array)):
        if flag_array[i] == 0:
            headpoint = [proparray[i][4],proparray[i][5]]
            image, headpoint_new = core_functions.get_neg_roi(filename, proparray[i], headpoint)
            if config.mod == True:
                image = mod.get_channels(image,headpoint_new)
            if image is not None:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                result, encimg = cv2.imencode('.jpg', image, encode_param)
                image = cv2.imdecode(encimg, 1)
                cv2.imwrite(neg_out_folder + filename_only_png.replace(".png", "_id_") + str(i) + ".png", image)

def main(_):
    data['frames'] = []
    rgb_files = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
    with open(config.gt_json) as f:
        jsongt = json.load(f)
    with open(config.prop_json) as f:
      jsonprop = json.load(f)
      
    for i in range(0,len(rgb_files)):
        process_image(rgb_folder + rgb_files[i], jsongt, jsonprop)
    with open(config.pos_out_json, 'w') as outfile:  
        json.dump(data, outfile, indent=4)
if __name__ == '__main__':
  tf.app.run()