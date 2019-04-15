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
import eval_functions as eval_functions
sys.path.append("../train_models/")
#from train_models.mtcnn_model import O_Net
import matplotlib.pyplot as plt
import logging
import singleton as singleton
import mod
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.FATAL)

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

        #val_folder = "/Data2TB/chl_data/mod/val/pos_neg_png_48/"
        images = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
        pos = 0
        neg = 0
        for image in images:
                img = cv2.imread(images_folder + image,-1)
                #img = cv2.imread(val_folder + "video_fifth_2018-04-23_CAM1_1524502510848_id_791_index_0.png")
                positive_props = []
                detections_regressed = []
                img = cv2.imread(images_folder + image,-1)
                #img = cv2.imread("/Data2TB/chl_data/mod/val/pos_png_48/video_fifth_2018-04-23_CAM1_1524502858816_id_831_index_0.png",-1)
                ##roi_detect = img.reshape(1, 48,48,3)
                ##roi_detect = (roi_detect.astype('float32') - 127.5)/128
                ##roi_detect = roi_detect[...,::-1]
                proposals, gtobjects = common_functions.get_proposal_and_gt_boxes_mod(image,jsonprop,jsongt)
                display_image = cv2.imread(display_folder + image)
                #
                #
                count = 0
                propcount = 0
                for prop in proposals:
                        total_detections += len(proposals)
                        imgtemp = img.copy()
                        #cv2.imshow("tmp",imgtemp)
                        #cv2.waitKey()
                        xmin = prop[0][0]
                        ymin = prop[0][1]
                        xmax = prop[0][2]
                        ymax = prop[0][3]

                        headpoint = prop[1]

                        xmin,ymin,xmax,ymax = common_functions.refine_bounding_box(xmin,ymin,xmax,ymax)

                        roi = imgtemp[ymin:ymax, xmin:xmax]

                        x_roi = headpoint[0] - xmin
                        y_roi = headpoint[1] - ymin

                        headpoint_roi = [y_roi,x_roi]

                        roi = mod.get_channels(roi,headpoint_roi)
                        #    #roi1copy = roi1.copy()
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        result, encimg = cv2.imencode('.jpg', roi, encode_param)
                        roi = cv2.imdecode(encimg, 1)

                        #cv2.imshow("WAS IT LOS", roi)
                        #cv2.waitKey()

                        count += 1
                        #    #continue
                        #    #videos_holland_orbbec_2018-02-21_CAM2_1519232343120_id_66320_index_0.png
                        width_original = roi.shape[1]
                        width_diff =  width_original - 48
                        #

                        roi_detect = cv2.resize(roi, (48,48))
                        roi_detect = roi_detect.reshape(1, 48,48,3)
                        roi_detect = (roi_detect.astype('float32') - 127.5)/128
                        roi_detect = roi_detect[...,::-1]
                        #cls_scores = common_functions.testpredict(roi_detect)

                        cls_scores = pingit.predict(roi_detect)[0]
                        #print(cls_scores)
                        ind = np.argmax(cls_scores)
                        if(ind == 1):
                                zabri = 1
                                pos += 1
                                #cv2.imshow("pos", img)
                                #cv2.waitKey()
                                positive_props.append(prop)
                                #bb,lm = common_functions.rescale_detections_to_full_image(bb,lm, prop)
                                #detection = [bb,lm]
                                #detections_regressed.append(detection)

                ious = eval_functions.get_iou_totals_per_image(gtobjects, positive_props,images_folder + image)
                ##shoulder_precisions = eval_functions.get_shoulder_precisions(gtobjects, detections_regressed)
                #
                iou_props_total += ious[0]
                iou_detections_total += ious[1]
                total_gt += ious[0]
                total_tp += ious[1]
                total_fp += ious[2]

                if(ious[0] != 0 and ious[1] != 0):
                        master_count += 1


        recall = total_tp/total_gt
        miss_rate = total_fp/(total_tp + total_fp)
        #av_pckh_precision = total_pckh_precision/(master_count)
        data['results'].append({
        'at' : str(index),
        'recall' : recall,
        'miss_rate' : miss_rate
        })
        with open(config.results_out + config.mtcnn_model + config.bbox_loss + ".json", 'a') as outfile:  
                json.dump(data, outfile, indent=4)
        return recall, miss_rate

def main(_):
    path = "/home/wassimea/Desktop/wassimea/work/train_models/mn"#sys.argv[1]
    sys.path.append(path)
    #config = __import__(sys.argv[2])
    config = __import__("MTCNN_config")
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
    index = 500#config.from_index
    while index <= 1500:#config.to_index:
        print("Processing index: ", index)
        ckpts_array.append(index)
        if(config.input_channels == 3):
            results = evaluate_3c(index)
        recall_array.append(results[0])
        missrate_array.append(results[1])
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
