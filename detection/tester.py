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
import singleton as singleton
import mtcnn_model as mtcnn_model
import os
import chl
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self):#, data_size, batch_size, model_path):
        net_factory = mtcnn_model.onet_cnn5

        model_path = '/Data2TB/chl_data/CKPTS/rgbd/mse/CNN5/4_onet_cnn5mse/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[None, data_size, data_size, 4], name='input_image')
            #figure out landmark            
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path + '-' + str(1490))
            for n in tf.get_default_graph().as_graph_def().node:
                print(n.name)
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

def evaluate_4c():
    times = []
    pingit = Detector()

    #d_n_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"
    d_folder = "/Data2TB/correctly_registered/S12/test/depth/"
    rgb_folder = "/Data2TB/correctly_registered/S12/test/color/"
    display_folder = "/Data2TB/correctly_registered/S12/test/color/"

    gt_path = "/Data2TB/sample/annotations.json"
    with open(gt_path) as f:
        jsondata_gt = json.load(f)
    png_files = [f for f in listdir(d_folder) if isfile(join(d_folder, f))]
    for image in png_files:
        gt_heads = common_functions.get_gt_annotations(image.replace(".png",".jpg"),jsondata_gt)
        rgb_img = cv2.imread(rgb_folder + image.replace(".png",".jpg"))
        display_img = cv2.imread(display_folder + image.replace(".png",".jpg"))
        d_img = cv2.imread(d_folder + image.replace(".jpg",".png"),-1)
        #d_n_img = cv2.imread(d_n_folder + image.replace(".jpg",".png"),cv2.IMREAD_GRAYSCALE)
        #startchl = time.clock()
        proposals = chl.findz(d_img)
        rgb_img = rgb_img[...,::-1]
        cv2.normalize(d_img,  d_img, 0, 255, cv2.NORM_MINMAX)
        d_img = d_img.reshape(d_img.shape[0],d_img.shape[1],1)
        combined_image = (np.concatenate((rgb_img, d_img), axis=2).astype('float32') - 127.5)/128
        #combined_image = (combined_image.astype('float32') - 127.5)/128
        for prop in proposals:
            #xmin = prop[0]
            #ymin = prop[1]
            #xmax = prop[2]
            #ymax = prop[3]
#
            roi_combined_image = combined_image[prop[1]:prop[3], prop[0]:prop[2]]
            #width_original = roi_combined_image.shape[1]
#
            roi_detect = cv2.resize(roi_combined_image, (48,48)).reshape(1,48,48,4)

            cls_scores, reg,landmark = pingit.predict(roi_detect)
            bb,lm = common_functions.rescale_detections_to_full_image(reg[0],landmark[0],prop)
            ind = np.argmax(cls_scores[0])
            if(ind == 1):
                cv2.rectangle(display_img,(prop[0], prop[1]), (prop[2], prop[3]),(255,0,0), 2)
                cv2.rectangle(display_img,(bb[0], bb[1]), (bb[2], bb[3]),(0,0,255), 2)
                cv2.circle(display_img,(lm[1],lm[0]),2,(0, 255, 255),thickness=2,lineType=8)
                cv2.circle(display_img,(lm[3],lm[2]),2,(0, 255, 255),thickness=2,lineType=8)
            #times.append(time.clock() - startchl)
        for gt in gt_heads:
            cv2.circle(display_img,(gt[4][0],gt[4][1]),2,(0, 255, 0),thickness=2,lineType=8)
            cv2.circle(display_img,(gt[5][0],gt[5][1]),2,(0, 255, 0),thickness=2,lineType=8)
            cv2.rectangle(display_img,(gt[0], gt[1]), (gt[2], gt[3]),(0,255,0), 2)
        cv2.imshow("im",display_img)
        cv2.waitKey()
        #times.append(time.clock() - startchl)
        #times.append(time.clock() - startchl)
    times = times[1:-1]
    print(times)
    print(np.mean(times))

def evaluate_3c():
    globe = []
    pingit = Detector()

    #d_n_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"
    d_folder = "/Data2TB/correctly_registered/S12/test/depth/"
    rgb_folder = "/Data2TB/correctly_registered/S12/test/color/"
    jpg_files = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
    for image in jpg_files:
        rgb_img = cv2.imread(rgb_folder + image)
        display_img = cv2.imread(rgb_folder + image)
        d_img = cv2.imread(d_folder + image.replace(".jpg",".png"),-1)
        #d_n_img = cv2.imread(d_n_folder + image.replace(".jpg",".png"),cv2.IMREAD_GRAYSCALE)
        start_chl = time.clock()
        proposals = chl.findz(d_img)
        rgb_img = rgb_img[...,::-1]
        #cv2.normalize(d_img,  d_img, 0, 255, cv2.NORM_MINMAX)
        #d_img = d_img.reshape(d_img.shape[0],d_img.shape[1],1)
        rgb_img = (rgb_img.astype('float32') - 127.5)/128
        #combined_image = (combined_image.astype('float32') - 127.5)/128
        ayre = []
        for prop in proposals:
            #xmin = prop[0]
            #ymin = prop[1]
            #xmax = prop[2]
            #ymax = prop[3]
            
            roi_rgb_img = rgb_img[prop[1]:prop[3], prop[0]:prop[2]]
            
            #width_original = roi_combined_image.shape[1]
            
            roi_detect = cv2.resize(roi_rgb_img, (48,48)).reshape(1,48,48,3)
            #ayre.append(roi_detect)
            
            cls_scores, reg,landmark = pingit.predict(roi_detect)
            
            bb,lm = common_functions.rescale_detections_to_full_image(reg[0],landmark[0],prop)
            ind = np.argmax(cls_scores[0])
            if(ind == 1):
                cv2.rectangle(display_img,(prop[0], prop[1]), (prop[2], prop[3]),(255,0,0), 1)
                cv2.rectangle(display_img,(bb[0], bb[1]), (bb[2], bb[3]),(0,0,255), 1)
                cv2.circle(display_img,(lm[1],lm[0]),2,(0,255,0),thickness=1,lineType=8)
                cv2.circle(display_img,(lm[3],lm[2]),2,(0,255,0),thickness=1,lineType=8)
            #times.append(time.clock() - startchl)
        cv2.imshow("im",display_img)
        cv2.waitKey()
        #x = np.array(ayre)
        #cls_scores, reg,landmark = pingit.predict(ayre)
        #for m in range(0,len(cls_scores)):
        #    bb,lm = common_functions.rescale_detections_to_full_image(reg[m],landmark[m],proposals[m])
        #    ind = np.argmax(cls_scores[m])
        #    if(ind == 1):
        #        cv2.rectangle(display_img,(bb[0], bb[1]), (bb[2], bb[3]),(0,0,255), 1)
        #globe.append(time.clock() - start_chl)
        #cv2.imshow("im",display_img)
        #cv2.waitKey()
        #times.append(time.clock() - startchl)
        #times.append(time.clock() - startchl)
        #times.append(time.clock() - startchl)
    globe = globe[1:-1]
    #print(times)
    print(str(1/(np.mean(globe)))," FPS")
    #print(sum(globe))

def main(_):
    #evaluate_3c()
    evaluate_4c()


if __name__ == '__main__':
  tf.app.run()
