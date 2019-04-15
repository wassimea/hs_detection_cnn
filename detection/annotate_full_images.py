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
import mtcnn_model as mtcnn_model


class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self,):#, data_size, batch_size, model_path):
        net_factory = mtcnn_model.onet_cnn4
        model_path = '/Data2TB/chl_data/CKPTS/rgbd/mse/CNN4/4_onet_cnn4mse/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 4], name='input_image')
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
            saver.restore(self.sess, "/Data2TB/chl_data/CKPTS/rgbd/mse/CNN4/4_onet_cnn4mse/-1160")                        

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
        
def get_bounding_box_WASSIMEA(x,y,z):
    factor = 1600 / z 
    xmin = int(x - (120 * factor / 2))
    ymin = int(y - (120 * (factor / 4)))
    xmax = int(xmin + (150 * factor))
    ymax = int(ymin + (150 * factor))
    return xmin, ymin, xmax, ymax

    return [xmin, ymin, xmax, ymax]


def get_proposal_boxes(filename):
    proposals = []
    with open("/Data2TB/correctly_registered/S12/test/output.json") as f:
        jsondata = json.load(f)
    filename_png = filename.replace(".jpg",".png")
    for jsonobj in jsondata:
        if jsonobj["file"] == filename_png:
            for prop in jsonobj["objects"]:
                x = prop["x"]
                y = prop["y"]
                z = prop["z"]
                box = get_bounding_box_WASSIMEA(x,y,z)
                proposals.append(box)
            break
    return proposals



def annotate():
    pingit = Detector()
    data ={}
    data['results'] = []
    rgb_folder = "/Data2TB/correctly_registered/S12/test/color/"
    d_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"
    rgb_images = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
    jsonprop = "/Data2TB/correctly_registered/S12/test/output.json"
    jsongt = "/Data2TB/sample/annotations.json"
    for image in rgb_images:
        rgb_img = cv2.imread(rgb_folder + image)
        d_img = cv2.imread(d_folder + image.replace(".jpg",".png"),cv2.IMREAD_GRAYSCALE)
        proposals, gtobjects = common_functions.get_proposal_and_gt_boxes(image,jsonprop,jsongt)
        for prop in proposals:
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
            cls_scores, reg,landmark = pingit.predict(roi_detect)
            bb = reg[0]
            lm = landmark[0]
            bb,lm = common_functions.rescale_detections_to_full_image(bb,lm,width_original, prop)
            ind = np.argmax(cls_scores[0])
            if(ind == 1):
                cv2.rectangle(rgb_img,(bb[0], bb[1]), (bb[2], bb[3]),(0,0,255), 1)
        cv2.imshow("image",rgb_img)
        cv2.waitKey()
def main(_):
    annotate()
if __name__ == '__main__':
  tf.app.run()
