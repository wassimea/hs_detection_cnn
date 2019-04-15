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
sys.path.append("../train_models/")
#from train_models.mtcnn_model import O_Net
import matplotlib.pyplot as plt
import logging
from scipy.spatial import distance
from itertools import cycle
from matplotlib.ticker import FuncFormatter

#import train_models.singleton as singleton
#import train_models.mtcnn_model as mtcnn_model
import singleton as singleton
import mtcnn_model as mtcnn_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.FATAL)


class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, model_path, index):#, data_size, batch_size, model_path):
        model_path_tmp = model_path[0:len(model_path)-1]
        model_def = os.path.basename(model_path_tmp)
        channels = int(model_def[0])
        onet_model = model_def.split('onet_')[1][0:4]
        if(onet_model == "cnn4"):
            net_factory = mtcnn_model.onet_cnn4
        elif(onet_model == "cnn5"):
            net_factory = mtcnn_model.onet_cnn5
        elif(onet_model == "cnn6"):
            net_factory = mtcnn_model.onet_cnn6

        #model_path = config.model_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + '/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, channels], name='input_image')
            #figure out landmark            
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            #model_dict = model_dict + '/'
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path + '-' + str(index))                        
            #saver.restore(self.sess, "/Data2TB/chl_data/CKPTS/CNN4/24102018/CNN4/-1084") 

        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self,image):
        #sess = tf.Session()
        #image = sess.run(image)
        #x = 1
        # access data
        # databatch: N x 3 x data_size x data_size
        #databatch = 1*3*48*48
        cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: image})
        return cls_prob, bbox_pred, landmark_pred


def evaluate():
    data ={}
    data['results'] = []
    models = []
    #models.append('/Data2TB/chl_data/CKPTS/rgb/mse/CNN4/3_onet_cnn4mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/mse/CNN5/3_onet_cnn5mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/mse/CNN6/3_onet_cnn6mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/mse/CNN5/3_onet_cnn5mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/mse/CNN6/3_onet_cnn6mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgbd/mse/CNN4/4_onet_cnn4mse/')
    #models.append('/Data2TB/chl_data/CKPTS/rgbd/mse/CNN5/4_onet_cnn5mse/')
    models.append('/Data2TB/chl_data/CKPTS/rgbd/mse/CNN6/4_onet_cnn6mse/')

    #models.append('/Data2TB/chl_data/CKPTS/rgb/ohem/CNN4/3_onet_cnn4ohem/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/ohem/CNN5/3_onet_cnn5ohem/')
    #models.append('/Data2TB/chl_data/CKPTS/rgb/ohem/CNN6/3_onet_cnn6ohem/')
    #models.append('/Data2TB/chl_data/CKPTS/rgbd/ohem/CNN4/4_onet_cnn4ohem/')
    #models.append('/Data2TB/chl_data/CKPTS/rgbd/ohem/CNN5/4_onet_cnn5ohem/')
    #models.append('/Data2TB/chl_data/CKPTS/rgbd/ohem/CNN6/4_onet_cnn6ohem/')

    thresholds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    #thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    percentages = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    recalls = []
    fps = []
    ious = []
    prop_ious = []
    pckhs = []
    #model_names = ['RGB_5_MSE','RGBD_5_MSE','AGG_5_0.4_MSE','AGG_5_0.6_MSE','AGG_5_0.8_MSE']
    model_names = ['RGB_4_MSE','RGB_5_MSE','RGB_6_MSE','RGBD_4_MSE','RGBD_5_MSE','RGBD_6_MSE']
    #model_names = ['RGB_4_ohem','RGB_5_ohem','RGB_6_ohem','RGBD_4_ohem','RGBD_5_ohem','RGBD_6_ohem']

    for model in models:
        folder = "/Data2TB/correctly_registered/S12/test/color/"
        model_path_tmp = model[0:-1]
        model_def = os.path.basename(model_path_tmp)
        channels = int(model_def[0])
        parent_folder = model[0: -15]
        with open(parent_folder + 'chosen.txt') as f:
            chosen = f.readlines()[0].replace('\n','')
        pingit = Detector(model,chosen)
        
        model_recalls = []
        model_ious = []
        model_fps = []
        model_pckhs = []
        model_prop_ious = []
        for threshold in thresholds:
            if(channels == 3):
                res = evaluate_3c(pingit,threshold,0.4,folder)
                model_ious.append(res[0])
                model_recalls.append(1 - res[1])
                model_fps.append(res[2])
                model_prop_ious.append(res[4])
                y = 1
            else:
                res = evaluate_4c(pingit,threshold,0.4)
                model_ious.append(res[0])
                model_recalls.append(1 - res[1])
                model_fps.append(res[2])
                model_prop_ious.append(res[4])
                y = 1
        #for percent in percentages:
        #    if(channels == 3):
        #        res = evaluate_3c(pingit,0.5,percent,folder)
        #        model_pckhs.append(res[3])
        #        y = 1
        #    else:
        #        res = evaluate_4c(pingit,0.5,percent)
        #        model_pckhs.append(res[3])
        #        y = 1
        recalls.append(model_recalls)
        fps.append(model_fps)
        ious.append(model_ious)
        pckhs.append(model_pckhs)
        prop_ious.append(model_prop_ious)
    #create_precision_recall_curve(fps,recalls,model_names)
    #create_bargraph(ious,prop_ious,model_names)
    #create_pckh_curve(pckhs,percentages,model_names)


def evaluate_3c(pingit,ayre,rabebe,images_folder):

    images = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
    jsonprop = "/Data2TB/correctly_registered/S12/test/output.json"
    jsongt = "/Data2TB/sample/annotations.json"

    #list_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]
    list_thresholds = [0.9]
    list_percentages = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    percentage = 0.1
    list_precisions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    list_recalls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    list_gt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]


    total_gt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_tn = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_tp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_fp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_outl = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_fn = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_pckh_precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_true_detections_p5 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    iou_props_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    iou_detections_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    master_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    av_iou_prop = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    av_iou_detection = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    miss_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    av_pckh_precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for image in images:
        total_proposals = 0
        #image = images[i]
        positive_props = []
        detections_regressed = []
        img = cv2.imread(images_folder + image)
        proposals, gtobjects = common_functions.get_proposal_and_gt_boxes(image,jsonprop,jsongt)

        count = 0
        propcount = 0

        list_positive_props = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        list_detections_regressed = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for prop in proposals:
            total_proposals += 1
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

            for i in range(len(list_percentages)):
                if(cls_scores[0][1] >= 0.9):
                    list_positive_props[i].append(prop)
                    bbtmp,lmtmp = common_functions.rescale_detections_to_full_image(bb,lm, prop)
                    list_detections_regressed[i].append([bbtmp,lmtmp])

        for i in range(len(list_percentages)):
            results = get_iou_totals_per_image(gtobjects, list_positive_props[i], list_detections_regressed[i],images_folder + image,list_percentages[i])
            iou_props_total[i] += results[0]
            iou_detections_total[i] += results[1]
            total_gt[i] += results[2]
            total_tp[i] += results[3]
            total_fp[i] += results[4]
            total_outl[i] += results[6]
            total_fn[i] += results[7]
            total_tn[i] += total_proposals - results[3] - results[4] - results[6]
            total_pckh_precision[i] += results[5]
            total_true_detections_p5[i] += results[8]
            #total_precisions += ious[5]
            if(results[0] != 0 and results[1] != 0):
                master_count[i] += 1

    for i in range(len(list_percentages)): 
        av_iou_prop[i] = iou_props_total[i]/master_count[i]
        av_iou_detection[i] = iou_detections_total[i]/master_count[i]
        recall[i] = total_tp[i] / total_gt[i]
        miss_rate[i] = total_fp[i] / (total_fp[i] + total_tn[i])
        precision[i] = total_tp[i] / (total_tp[i] + total_fp[i])
        av_pckh_precision[i] = total_pckh_precision[i]/(master_count[i])
        #recall[i] = total_true_detections_p5[i] / total_gt[i]
        #precision[i] = total_true_detections_p5[i] / (total_true_detections_p5[i] + total_fp[i])

        #print("Threshold: " , round(list_thresholds[i],2))
        #print("Percentage: " , round(list_percentages[i],2))
        #print("Average prop IOU: " , round(av_iou_prop[i],2))
        #print("Average detection IOU: " , round(av_iou_detection[i],2))
        #print("PCKH: " , round(av_pckh_precision[i],2))
        #print("Recall: " , round(recall[i],2))
        #print("Precision: " , round(precision[i],2))

        #print("")
        #print("")
        print("Average PCKH: " , round(av_pckh_precision[i],2))
    print("Average prop IOU: " , round(av_iou_prop[13],2))
    print("Average detection IOU: " , round(av_iou_detection[13],2))

    return av_iou_detection,recall, miss_rate,av_pckh_precision,av_iou_prop

def evaluate_4c(pingit,ayre,images_folder):
    
    rgb_folder = "/Data2TB/correctly_registered/S12/test/color/"
    images_folder = rgb_folder
    d_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"
    display_folder = "/Data2TB/correctly_registered/S12/test/color/"
    jsonprop = "/Data2TB/correctly_registered/S12/test/output.json"
    jsongt = "/Data2TB/sample/annotations.json"

    rgb_images = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
    jsonprop = "/Data2TB/correctly_registered/S12/test/output.json"
    jsongt = "/Data2TB/sample/annotations.json"

    #list_thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9]
    list_thresholds = [0.9]
    list_percentages = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    percentage = 0.5
    list_precisions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    list_recalls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    list_gt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]


    total_gt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_tn = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_tp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_fp = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_outl = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_fn = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_pckh_precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    total_true_detections_p5 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    iou_props_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    iou_detections_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    master_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    av_iou_prop = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    av_iou_detection = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    recall = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    miss_rate = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    av_pckh_precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for image in rgb_images:
        total_proposals = 0
        #image = images[i]
        positive_props = []
        detections_regressed = []
        rgb_img = cv2.imread(rgb_folder + image)
        d_img = cv2.imread(d_folder + image.replace(".jpg",".png"),cv2.IMREAD_GRAYSCALE)
        proposals, gtobjects = common_functions.get_proposal_and_gt_boxes(image,jsonprop,jsongt)

        count = 0
        propcount = 0

        list_positive_props = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        list_detections_regressed = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for prop in proposals:
            total_proposals += 1

            imgtemp_rgb = rgb_img.copy()
            imgtemp_rgb = imgtemp_rgb[...,::-1]
            imgtemp_d = d_img.copy()
            imgtemp_d = imgtemp_d.reshape(imgtemp_d.shape[0],imgtemp_d.shape[1],1)
            combined_image = np.concatenate((imgtemp_rgb, imgtemp_d), axis=2)
            #cv2.imshow("tmp",imgtemp)
            #cv2.waitKey()
            xmin = prop[0]
            ymin = prop[1]
            xmax = prop[2]
            ymax = prop[3]
            xmin,ymin,xmax,ymax = common_functions.refine_bounding_box(xmin,ymin,xmax,ymax)

            roi_combined_image = combined_image[ymin:ymax, xmin:xmax]
            #roi_rgb = imgtemp_rgb[ymin:ymax, xmin:xmax]

            width_original = roi_combined_image.shape[1]
            width_diff =  width_original - 48

            roi_detect = cv2.resize(roi_combined_image, (48,48))
            roi_detect = roi_detect.reshape(1,48,48,4)
            roi_detect = (roi_detect.astype('float32') - 127.5)/128

            #roi_detect_rgb = cv2.resize(roi_rgb, (48,48))
            #roi_detect_rgb = roi_detect_rgb.reshape(48,48,3)
            #roi_detect_rgb = (roi_detect_rgb.astype('float32') - 127.5)/128
            #roi_detect_rgb = roi_detect_rgb[...,::-1]

            cls_scores, reg,landmark = pingit.predict(roi_detect)

            bb = reg[0]
            lm = landmark[0]

            for i in range(len(list_percentages)):
                if(cls_scores[0][1] >= 0.9):
                    list_positive_props[i].append(prop)
                    bbtmp,lmtmp = common_functions.rescale_detections_to_full_image(bb,lm, prop)
                    list_detections_regressed[i].append([bbtmp,lmtmp])

        for i in range(len(list_percentages)):
            results = get_iou_totals_per_image(gtobjects, list_positive_props[i], list_detections_regressed[i],images_folder + image,list_percentages[i])
            iou_props_total[i] += results[0]
            iou_detections_total[i] += results[1]
            total_gt[i] += results[2]
            total_tp[i] += results[3]
            total_fp[i] += results[4]
            total_outl[i] += results[6]
            total_fn[i] += results[7]
            total_tn[i] += total_proposals - results[3] - results[4] - results[6]
            total_pckh_precision[i] += results[5]
            total_true_detections_p5[i] += results[8]
            #total_precisions += ious[5]
            if(results[0] != 0 and results[1] != 0):
                master_count[i] += 1

    for i in range(len(list_percentages)): 
        av_iou_prop[i] = iou_props_total[i]/master_count[i]
        av_iou_detection[i] = iou_detections_total[i]/master_count[i]
        recall[i] = total_tp[i] / total_gt[i]
        miss_rate[i] = total_fp[i] / (total_fp[i] + total_tn[i])
        precision[i] = total_tp[i] / (total_tp[i] + total_fp[i])
        av_pckh_precision[i] = total_pckh_precision[i]/(master_count[i])
        #recall[i] = total_true_detections_p5[i] / total_gt[i]
        #precision[i] = total_true_detections_p5[i] / (total_true_detections_p5[i] + total_fp[i])

        #print("Threshold: " , round(list_thresholds[i],2))
        #print("Average prop IOU: " , round(av_iou_prop[i],2))
        #print("Average detection IOU: " , round(av_iou_detection[i],2))
        #print("PCKH: " , round(av_pckh_precision[i],2))
        #print("Recall: " , round(recall[i],2))
        #print("Precision: " , round(precision[i],2))
        print("Average PCKH: " , round(av_pckh_precision[i],2))
        #print("")
        #print("")

    #print("Average prop IOU: " , round(av_iou_prop[13],2))
    #print("Average detection IOU: " , round(av_iou_detection[13],2))
    #print("Average PCKH: " , round(av_pckh_precision[6],2))
    return av_iou_detection,recall, miss_rate,av_pckh_precision,av_iou_prop

def get_iou_totals_per_image(gt_objects, positive_props,detections_regressed,file,percentage):
    total_props_iou = 0
    total_detections_iou = 0
    total_positives = 0
    total_negatives = 0
    total_gt = 0
    total_pckh_precisions = 0
    outliers = 0
    total_fn = 0
    total_true_detections_p5 = 0
    list_found_indices = []
    for gt in gt_objects:
        if(common_functions.check_box_boundaries(gt) == True):
            total_gt += 1
            for prop in positive_props:
                #if(common_functions.check_if_rectangle_contains_another(prop,gt)):
                if(common_functions.bb_intersection_over_boxA(gt,prop) >= 0.5):
                    total_positives += 1
                    list_found_indices.append(positive_props.index(prop))
                    total_props_iou += common_functions.bb_intersection_over_union(prop,gt)
                    prop_index = positive_props.index(prop)

                    detection_iou = common_functions.bb_intersection_over_union(detections_regressed[prop_index][0],gt)
                    total_detections_iou += detection_iou
                    if detection_iou > 0.5:
                        total_true_detections_p5 += 1
                    pckh_precision = get_shoulder_precisions(gt,detections_regressed[prop_index][1],file,percentage)
                    total_pckh_precisions += pckh_precision
                    break
    for i in range (0,len(positive_props)):
        if not i in list_found_indices:
            if(common_functions.check_box_boundaries(positive_props[i]) == True):
                total_negatives += 1
            else:
                outliers += 1

    #total_negatives = len(positive_props) - total_positives
    average_props_iou = 0
    average_detections_iou = 0
    average_pckh_precision = 0
    if(total_positives != 0):
        average_props_iou = total_props_iou/total_positives
        average_detections_iou = total_detections_iou/total_positives
        average_pckh_precision = total_pckh_precisions/(total_positives * 2)
    test = total_gt - (total_positives + total_negatives)
    if test > 0:
        total_fn = test
    return average_props_iou, average_detections_iou, total_gt, total_positives, total_negatives,average_pckh_precision, outliers,total_fn, total_true_detections_p5

def get_shoulder_precisions(gt, lm,file,percentage):
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

    threshold = percentage * max(width,height)

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

def main(_):
    tf.logging.set_verbosity(tf.logging.FATAL)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    evaluate()

if __name__ == '__main__':
  tf.app.run()
