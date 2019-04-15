from skimage.io import imread
from skimage.io import imsave
from skimage.filters import rank
from skimage.morphology import rectangle
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
import random
#sys.path.append("../train_models/")
import mtcnn_model as mtcnn_model




class Detector(object):

    def __init__(self):
        net_factory = mtcnn_model.onet_cnn4

        model_path = '/Data2TB/chl_data/CKPTS/rgb/mse/CNN4/3_onet_cnn4mse/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[None, data_size, data_size, 3], name='input_image')
            #figure out landmark            
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether or not the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path + '-' + str(1100))
            #for n in tf.get_default_graph().as_graph_def().node:
            #    print(n.name)
            #saver.restore(self.sess, "/Data2TB/chl_data/CKPTS/CNN4/24102018/CNN4/-1084") 
        self.data_size = data_size
        self.batch_size = batch_size
    def predict(self,image):
        cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: image})
        return cls_prob, bbox_pred, landmark_pred

def generate():
    pingit = Detector()
    parent_folder = "/Data2TB/andres_dataset/annotated/temp/"
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir() ]
    for subfolder in subfolders:
        customers = [f.path for f in os.scandir(subfolder) if f.is_dir() ]
        #out_folder = "/Data2TB/andres_dataset/"
        for customer in customers:
            files = [f for f in listdir(customer) if isfile(join(customer, f))]
            for image in files:
                if ".png" in image:
                    print(customer + "/" + image)
                    if(os.path.exists(customer + "/" + image.replace(".png",".json"))):
                        with open(customer + "/" + image.replace(".png",".json")) as f:
                            oldData = json.load(f)
                        img = cv2.imread(customer + "/" + image,cv2.IMREAD_UNCHANGED)
                        img[:, :, 3] = 255
                        img = img[:,:,:3]


                        width = img.shape[1]

                        roi = img[0:width, 0:width]
                        roi = roi[...,::-1]
                        roi = (roi.astype('float32') - 127.5)/128
                        roi = cv2.resize(roi, (48,48))
                        roi = roi.reshape(1,48,48,3)

                        cls_scores, reg,landmark = pingit.predict(roi)
                        bb,lm = rescale_detections_to_full_image(reg[0],landmark[0],[width,width])
                        head_data = {}
                        s1_data = {}
                        s2_data = {}

                        head_data["head"] = {'confidence' : str(cls_scores[0][1]), 'xmin' : bb[0], 'ymin' : bb[1], 'xmax' : bb[2], 'ymax' : bb[3]}
                        s1_data["shoulder_1"] = {'y' : lm[1], 'x' : lm[0]}
                        s2_data["shoulder_2"] = {'y' : lm[3], 'x' : lm[2]}

                        oldData.update(head_data)
                        oldData.update(s1_data)
                        oldData.update(s2_data)
                        with open(customer + "/" + image.replace(".png",".json"), 'w') as outfile:  
                            json.dump(oldData, outfile, indent=4)


                        #ind = np.argmax(cls_scores[0])
                        #imcpy = img.copy()
                        #if(ind == 1):
                        #    cv2.rectangle(imcpy,(bb[0], bb[1]), (bb[2], bb[3]),(0,0,255), 1)
                        #    cv2.circle(imcpy,(lm[1],lm[0]), 3, (0,255,0), -1)
                        #    cv2.circle(imcpy,(lm[3],lm[2]), 3, (0,255,0), -1)
                        #    #cv2.imwrite(out_folder + image,imcpy)
                        #print(imcpy.shape)
                        #cv2.imshow("imcpy",imcpy)
                        #cv2.waitKey()


def rescale_detections_to_full_image(bb,lm, propbox):
    width_original = propbox[0]#propbox[2] - propbox[0]
    prop48_xmin = int(bb[0] * 48)
    prop48_ymin = int(bb[1] * 48)
    prop48_xmax = int(bb[2] * 48)
    prop48_ymax = int(bb[3] * 48)

    prop48_xls = int(lm[0] * 48)
    prop48_yls = int(lm[1] * 48)
    prop48_xrs = int(lm[2] * 48)
    prop48_yrs = int(lm[3] * 48)

    prop_xmin = int((width_original * prop48_xmin) / 48)
    prop_ymin = int((width_original * prop48_ymin) / 48)
    prop_xmax = int((width_original * prop48_xmax) / 48)
    prop_ymax = int((width_original * prop48_ymax) / 48)

    prop_xls = int((width_original * prop48_xls) / 48)
    prop_yls = int((width_original * prop48_yls) / 48)
    prop_xrs = int((width_original * prop48_xrs) / 48)
    prop_yrs = int((width_original * prop48_yrs) / 48)



    #below lines are just to make sure...

    if prop_xmin < 0:               
        prop_xmin = 0
    if prop_ymin < 0:
        prop_ymin = 0
    if prop_xmax > width_original:
        prop_xmax = width_original
    if prop_ymax > width_original:
        prop_ymax = width_original

    if prop_xls < 0:
        prop_xls = 0
    if prop_yls < 0:
        prop_yls = 0
    if prop_xls > width_original:
        prop_xls = width_original
    if prop_yls > width_original:
        prop_yls = width_original

    if prop_xrs < 0:
        prop_xrs = 0
    if prop_yrs < 0:
        prop_yrs = 0
    if prop_xrs > width_original:
        prop_xrs = width_original
    if prop_yrs > width_original:
        prop_yrs = width_original
    
    bb = [prop_xmin,prop_ymin,prop_xmax,prop_ymax]
    lm = [prop_xls,prop_yls,prop_xrs,prop_yrs]

    return bb, lm    





def main(_): 
  generate()

if __name__ == '__main__':
  tf.app.run()
