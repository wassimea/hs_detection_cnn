import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
from os import listdir
from os.path import isfile, join
import nms as nms
import sys
import common_functions as common_functions
import eval_functions as eval_functions
sys.path.append("../train_models/")
#from train_models.mtcnn_model import O_Net
import matplotlib.pyplot as plt
import logging
import train_models.singleton as singleton

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.FATAL)

def evaluate_3c(index):
    config = singleton.configuration._instance.config
    data ={}
    data['results'] = []
    tf.logging.set_verbosity(tf.logging.FATAL)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    pingit = common_functions.Detector(index,config)
    images_folder = config.rgb_folder
    display_folder = config.display_folder
    images = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
    jsonprop = config.jsonprop
    jsongt = config.jsongt

    iou_props_total = 0
    iou_detections_total = 0

    total_detections = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_pckh_precision = 0
    master_count = 0

    for image in images:
        positive_props = []
        detections_regressed = []
        img = cv2.imread(images_folder + image)

        proposals, gtobjects = common_functions.get_proposal_and_gt_boxes(image,jsonprop,jsongt)
        display_image = cv2.imread(display_folder + image)


        count = 0
        propcount = 0
        for prop in proposals:
            total_detections += len(proposals)
            imgtemp = img.copy()
            #cv2.imshow("tmp",imgtemp)
            #cv2.waitKey()
            xmin = prop[0]
            ymin = prop[1]
            xmax = prop[2]
            ymax = prop[3]

            xmin,ymin,xmax,ymax = common_functions.refine_bounding_box(xmin,ymin,xmax,ymax)

            roi = imgtemp[ymin:ymax, xmin:xmax]

            width_original = roi.shape[1]
            width_diff =  width_original - 48

            roi_detect = cv2.resize(roi, (48,48))
            roi_detect = roi_detect.reshape(1, 48,48,3)
            roi_detect = (roi_detect.astype('float32') - 127.5)/128
            roi_detect = roi_detect[...,::-1]

            cls_scores, reg,landmark = pingit.predict(roi_detect)

            bb = reg[0]
            lm = landmark[0]

            ind = np.argmax(cls_scores[0])
            if(ind == 1):
                positive_props.append(prop)
                bb,lm = common_functions.rescale_detections_to_full_image(bb,lm,width_original, prop)
                detection = [bb,lm]
                detections_regressed.append(detection)
        
        ious = eval_functions.get_iou_totals_per_image(gtobjects, positive_props, detections_regressed,images_folder + image)
        #shoulder_precisions = eval_functions.get_shoulder_precisions(gtobjects, detections_regressed)

        iou_props_total += ious[0]
        iou_detections_total += ious[1]
        total_gt += ious[2]
        total_tp += ious[3]
        total_fp += ious[4]
        total_pckh_precision += ious[5]
        #total_precisions += ious[5]
        if(ious[0] != 0 and ious[1] != 0):
            master_count += 1
    
    av_iou_prop = iou_props_total/master_count
    av_iou_detection = iou_detections_total/master_count
    recall = total_tp/total_gt
    miss_rate = total_fp/(total_tp + total_fp)
    av_pckh_precision = total_pckh_precision/(master_count)
    data['results'].append({
                'at' : str(index),
                'recall' : recall,
                'miss_rate' : miss_rate,
                'av_iou' : av_iou_detection,
                'pckh' : av_pckh_precision
            })
    with open(config.results_out + config.mtcnn_model + config.bbox_loss + ".json", 'a') as outfile:  
        json.dump(data, outfile, indent=4)
    return av_iou_detection,recall, miss_rate

def evaluate_4c(index):
    config = singleton.configuration._instance.config
    data ={}
    data['results'] = []
    tf.logging.set_verbosity(tf.logging.FATAL)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    pingit = common_functions.Detector(index,config)
    rgb_folder = config.rgb_folder
    d_folder = config.d_folder
    display_folder = config.display_folder
    jsonprop = config.jsonprop
    jsongt = config.jsongt

    rgb_images = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]

    iou_props_total = 0
    iou_detections_total = 0

    total_detections = 0
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_pckh_precision = 0
    master_count = 0

    gt_path = "/Data2TB/sample/annotations.json"

    for image in rgb_images:
        positive_props = []
        detections_regressed = []
        rgb_img = cv2.imread(rgb_folder + image)
        d_img = cv2.imread(d_folder + image.replace(".jpg",".png"),cv2.IMREAD_GRAYSCALE)
        proposals, gtobjects = common_functions.get_proposal_and_gt_boxes(image,jsonprop,jsongt)
        display_image = cv2.imread(display_folder + image)


        count = 0
        propcount = 0
        for prop in proposals:
            total_detections += len(proposals)
            imgtemp_rgb = rgb_img.copy()
            imgtemp_d = d_img.copy()
            #cv2.imshow("tmp",imgtemp)
            #cv2.waitKey()
            xmin = prop[0]
            ymin = prop[1]
            xmax = prop[2]
            ymax = prop[3]

            xmin,ymin,xmax,ymax = common_functions.refine_bounding_box(xmin,ymin,xmax,ymax)

            roi_rgb = imgtemp_rgb[ymin:ymax, xmin:xmax]

            width_original = roi_rgb.shape[1]
            width_diff =  width_original - 48

            roi_detect_rgb = cv2.resize(roi_rgb, (48,48))
            roi_detect_rgb = roi_detect_rgb.reshape(48,48,3)
            roi_detect_rgb = (roi_detect_rgb.astype('float32') - 127.5)/128
            roi_detect_rgb = roi_detect_rgb[...,::-1]

            roi_d = imgtemp_d[ymin:ymax, xmin:xmax]

            width_original = roi_d.shape[1]
            width_diff =  width_original - 48

            #r,g,b = roi_d.split()
            roi_detect_d = cv2.resize(roi_d, (48,48))
            roi_detect_d = roi_detect_d.reshape(48,48,1)
            roi_detect_d = (roi_detect_d.astype('float32') - 127.5)/128
            #roi_detect = roi_detect[...,::-1]

            #roi_detect = np.dstack((roi_detect_rgb,roi_detect_d))
            roi_detect = np.concatenate((roi_detect_rgb, roi_detect_d), axis=2)
            roi_detect = roi_detect.reshape(1,48,48,4)
            #roi_detect = (roi_detect.astype('float32') - 127.5)/128
            cls_scores, reg,landmark = pingit.predict(roi_detect)

            bb = reg[0]
            lm = landmark[0]

            ind = np.argmax(cls_scores[0])
            if(ind == 1):
                positive_props.append(prop)
                bb,lm = common_functions.rescale_detections_to_full_image(bb,lm,width_original, prop)
                detection = [bb,lm]
                detections_regressed.append(detection)
        
        ious = eval_functions.get_iou_totals_per_image(gtobjects, positive_props, detections_regressed,rgb_folder + image)
        #shoulder_precisions = eval_functions.get_shoulder_precisions(gtobjects, detections_regressed)

        iou_props_total += ious[0]
        iou_detections_total += ious[1]
        total_gt += ious[2]
        total_tp += ious[3]
        total_fp += ious[4]
        #total_precisions += ious[5]
        total_pckh_precision += ious[5]
        if(ious[0] != 0 and ious[1] != 0):
            master_count += 1
    
    av_iou_prop = iou_props_total/master_count
    av_iou_detection = iou_detections_total/master_count
    recall = total_tp/total_gt
    miss_rate = total_fp/(total_tp + total_fp)
    av_pckh_precision = total_pckh_precision/(master_count)
    data['results'].append({
                'at' : str(index),
                'recall' : recall,
                'miss_rate' : miss_rate,
                'av_iou' : av_iou_detection,
                'pckh' : av_pckh_precision
            })
    with open(config.results_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + ".json", 'a') as outfile:  
        json.dump(data, outfile, indent=4)
    return av_iou_detection,recall, miss_rate

def main(_):
    path = sys.argv[1]
    sys.path.append(sys.argv[1])
    config = __import__(sys.argv[2])
    x = singleton.configuration(config.config)
    config = config.config
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.eval_gpu
    f= open(config.results_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + ".json","w+")
    f.close()
    #annotate()
    recall_array = []
    missrate_array = []
    iou_array = []
    ckpts_array = []
    index = config.from_index
    while index <= config.to_index:
        print("Processing index: ", index)
        ckpts_array.append(index)
        if(config.input_channels == 3):
            results = evaluate_3c(index)
        else:
            results = evaluate_4c(index)
        iou_array.append(results[0])
        recall_array.append(results[1])
        missrate_array.append(results[2])
        index += config.step_index
    #pyplot.plot(np.array(missrate_array),np.array(recall_array))
    #pyplot.ylabel('recall')
    #pyplot.xlabel('missrate')

    plt.plot(ckpts_array, recall_array, 'g') # plotting t, a separately 
    plt.plot(ckpts_array, missrate_array, 'r') # plotting t, b separately 
    plt.savefig(config.results_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + ".png")
    #plt.show()
    #pyplot.xticks(np.range(0,1))
    #pyplot.axis(np.array(ckpts_array))
    #pyplot.show()
    x = 1


if __name__ == '__main__':
  tf.app.run()
