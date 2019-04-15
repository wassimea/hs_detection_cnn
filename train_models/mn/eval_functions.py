import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
from os import listdir
from os.path import isfile, join
#import nms as nms
import sys
import common_functions as common_functions
sys.path.append("../")
#from train_models.mtcnn_model import O_Net
from scipy.spatial import distance  

def get_iou_totals_per_image(gt_objects, positive_props,file):
    total_props_iou = 0
    total_detections_iou = 0
    total_positives = 0
    total_negatives = 0
    total_gt = 0
    total_pckh_precisions = 0
    list_found_indices = []
    for gt in gt_objects:
        if(common_functions.check_box_boundaries(gt) == True):
            total_gt += 1
            for prop in positive_props:
                if(common_functions.check_if_rectangle_contains_another(prop[0],gt)):
                    list_found_indices.append(positive_props.index(prop))
                    total_positives += 1
                    #total_props_iou += common_functions.bb_intersection_over_union(prop,gt)
                    prop_index = positive_props.index(prop)
                    #total_detections_iou += common_functions.bb_intersection_over_union(detections_regressed[prop_index][0],gt)
                    #pckh_precision = get_shoulder_precisions(gt,detections_regressed[prop_index][1],file)
                    #total_pckh_precisions += pckh_precision
                    break
    for i in range (0,len(positive_props)):
        if not i in list_found_indices:
            if(common_functions.check_box_boundaries(positive_props[i][0]) == True):
                total_negatives += 1

    #total_negatives = len(positive_props) - total_positives
    #average_props_iou = 0
    #average_detections_iou = 0
    #average_pckh_precision = 0
    #if(total_positives != 0):
    #    average_props_iou = total_props_iou/total_positives
    #    average_detections_iou = total_detections_iou/total_positives
    #    average_pckh_precision = total_pckh_precisions/(total_positives * 2)

    return total_gt, total_positives, total_negatives

def get_shoulder_precisions(gt, lm,file):
    pckh_precisions = 0
    gts1 = gt[4]
    gts2 = gt[5]
    dets1 = (lm[1],lm[0])
    dets2 = (lm[3],lm[2])

    xmin = gt[0]
    ymin = gt[1]
    xmax = gt[2]
    ymax = gt[3]

    width = xmax - xmin
    height = ymax - ymin

    threshold = 0.4 * max(width,height)

    gt_point1 = gts1
    gt_point2 = gts2
    if(gts2[0] < gts1[0]):
        gt_point1 = gts2
        gt_point2 = gts1
    
    det_point1 = dets1
    det_point2 = dets2
    if(dets2[0]<dets1[0]):
        det_point1 = dets2
        det_point2 = dets1
    distance1 = distance.euclidean(gt_point1, det_point1)
    distance2 = distance.euclidean(gt_point2,det_point2)
    
    if(distance1 <= threshold):
        pckh_precisions += 1
    if(distance2 <= threshold):
        pckh_precisions += 1
    return pckh_precisions