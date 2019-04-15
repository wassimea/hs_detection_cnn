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
sys.path.append("../train_models/")
#import singleton as singleton
#import mtcnn_model as mtcnn_model
#import train_models.singleton as singleton
#import train_models.mtcnn_model as mtcnn_model
class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, index,config):#, data_size, batch_size, model_path):
        if(config.mtcnn_model == "onet_cnn4"):
            net_factory = mtcnn_model.onet_cnn4
        elif(config.mtcnn_model == "onet_cnn5"):
            net_factory = mtcnn_model.onet_cnn5
        elif(config.mtcnn_model == "onet_cnn6"):
            net_factory = mtcnn_model.onet_cnn6

        model_path = config.model_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + '/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, config.input_channels], name='input_image')
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
            saver.restore(self.sess, config.model_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + '/' + '-' + str(index))                        
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


def get_bounding_box_WASSIMEA(x,y,z):
    factor = 1600 / z 
    xmin = int(x - (120 * factor / 2))
    ymin = int(y - (120 * (factor / 4)))
    xmax = int(xmin + (150 * factor))
    ymax = int(ymin + (150 * factor))
    return xmin, ymin, xmax, ymax

    return [xmin, ymin, xmax, ymax]


def get_proposal_and_gt_boxes(filename,jsonprop,jsongt):
    proposals = []
    with open(jsonprop) as f:
        jsondata_prop = json.load(f)
    filename_png = filename.replace(".jpg",".png")
    for jsonobj in jsondata_prop:
        if jsonobj["file"] == filename_png:
            for prop in jsonobj["objects"]:
                x = prop["x"]
                y = prop["y"]
                z = prop["z"]
                box = get_bounding_box_WASSIMEA(x,y,z)
                proposals.append(box)
            break

    filename_jpg = filename.replace(".png",".jpg")
    #filename_jpg = filename_png
    with open(jsongt) as f:
        jsondata_gt = json.load(f)

    gtarray = []
    right_shoulder = [-1,-1]
    left_shoulder = [-1,-1]
    if filename_jpg in jsondata_gt and len(jsondata_gt[filename]["annotations"]) > 0:
        for object_gt in jsondata_gt[filename_jpg]["annotations"]:
            if object_gt["category"] == "Head":
                xmingt = object_gt["x"] 
                ymingt = object_gt["y"]
                width = object_gt["width"]
                height = object_gt["height"]
                xmaxgt = xmingt + width
                ymaxgt = ymingt + height
                id = object_gt["id"]
                for shoulder_candidate in jsondata_gt[filename]["annotations"]:
                    if shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Right Shoulder":
                        right_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                    elif shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Left Shoulder":
                        left_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                gtarray.append([xmingt,ymingt, xmaxgt,ymaxgt,right_shoulder,left_shoulder, id])
    return proposals, gtarray

def get_gt_annotations(filename, jsongt):
    gtarray = []
    right_shoulder = [-1,-1]
    left_shoulder = [-1,-1]
    if filename in jsongt and len(jsongt[filename]["annotations"]) > 0:
        for object_gt in jsongt[filename]["annotations"]:
            if object_gt["category"] == "Head":
                xmingt = object_gt["x"] #+ 5
                ymingt = object_gt["y"]
                width = object_gt["width"]
                height = object_gt["height"]
                xmaxgt = xmingt + width
                ymaxgt = ymingt + height
                id = object_gt["id"]
                for shoulder_candidate in jsongt[filename]["annotations"]:
                    if shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Right Shoulder":
                        right_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                    elif shoulder_candidate["id"] == id and shoulder_candidate["category"] == "Left Shoulder":
                        left_shoulder = [shoulder_candidate["x"],shoulder_candidate["y"]]
                gtarray.append([xmingt,ymingt, xmaxgt,ymaxgt,right_shoulder,left_shoulder, id])
    return gtarray

def refine_bounding_box(xmin,ymin,xmax,ymax):
    if(xmin < 0):
        xmin = 0
    if(ymin < 0):
        ymin = 0
    if(xmax > 640):
        xmax = 640
    if(ymax > 480):
        ymax = 480
    width = xmax - xmin
    height = ymax - ymin
    while (height != width):
        if(width > height):
            ymax = ymax + 1
            height = ymax - ymin
        elif(height > width):
            xmax = xmax + 1
            width = xmax - xmin
    return xmin,ymin,xmax,ymax


def rescale_detections_to_full_image(bb,lm, propbox):
    width_original = propbox[2] - propbox[0]
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


    img_xmin = propbox[0] + prop_xmin
    img_ymin = propbox[1] + prop_ymin
    img_xmax = img_xmin + (prop_xmax - prop_xmin)
    img_ymax = img_ymin + (prop_ymax - prop_ymin) 

    img_xls = propbox[1] + prop_xls
    img_yls = propbox[0] + prop_yls
    img_xrs = propbox[1] + prop_xrs
    img_yrs = propbox[0] + prop_yrs

    bb = [img_xmin,img_ymin,img_xmax,img_ymax]
    lm = [img_xls,img_yls,img_xrs,img_yrs]

    return bb, lm

def check_box_boundaries(gt):
    xmin = gt[0]
    ymin = gt[1]
    xmax = gt[2]
    ymax = gt[3]
    if xmin > 48 and ymin > 30 and xmax < 550 and ymax < 415:
        return True
    else:
        return False

def check_if_rectangle_contains_another(rect1, rect2):

    xmin1 = rect1[0]
    ymin1 = rect1[1]
    xmax1 = rect1[2]
    ymax1 = rect1[3]

    xmin2 = rect2[0]
    ymin2 = rect2[1]
    xmax2 = rect2[2]
    ymax2 = rect2[3]

    if(xmin1 < xmin2 and ymin1 < ymin2 and xmax1 > xmax2 and ymax1 > ymax2):
        return True
    else:
        return False
    
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou
def bb_intersection_over_boxA(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea)# + boxBArea - interArea)
 
	# return the intersection over union value
	return iou