import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
from os import listdir
from os.path import isfile, join
import functions.core_functions as core_functions


camera_factor = 1
camera_cx = 325.5
camera_cy = 253.5
camera_fx = 518.0
camera_fy = 519.0



class HeadClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = '/Data2TB/frozen/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        config = tf.ConfigProto(device_count = {'GPU': 1})
        self.sess = tf.Session(config=config,graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        x = len(boxes)
        arrb = boxes
        return arrb, scores, classes, num
        
#function that evaluates the accuracy of the model
def evaluate():
    with tf.device('/cpu:0'):   #run on CPU
        pingit = HeadClassifier()
    folder = "/home/wassimea/Desktop/SMATS/images/8bit/test/"   #images folder
    with open("/home/wassimea/Desktop/SMATS/combined.json") as f:   #open json file containing annotations
        jsondata = json.load(f)
    images = [f for f in listdir(folder) if isfile(join(folder, f))]
    list_results = [0,0,0,0.,0,0,0,0,0,0,0,0]
    list_total_pos = [0,0,0,0.,0,0,0,0,0,0,0,0]
    list_total_neg = [0,0,0,0.,0,0,0,0,0,0,0,0]
    list_binary_pos = [0,0,0,0.,0,0,0,0,0,0,0,0]
    list_binary_neg = [0,0,0,0.,0,0,0,0,0,0,0,0]
    list_thresholds = [0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.1,0.15,0.2,0.25,0.3] #list of thresholds to be tested
    for k in range(0,len(images)):
        if("resized-1-rotated-0" in images[k]):
            gtHeadCount = 0
            for i in range(0,len(jsondata['frames'])):
                #print(jsondata['frames'][i]['file'])
                if(jsondata['frames'][i]['file'] == images[k]):
                    for j in range(0,len(jsondata['frames'][i]['annotations'])):
                        if(jsondata['frames'][i]['annotations'][j]['label'] == 'Head'):
                            gtHeadCount += 1
                    break
            print(images[k])
            for j in range(0,len(list_thresholds)):
                #img = cv2.imread(folder + images[k])
                img = np.array(Image.open(folder + images[k]))
                new_img = img
                boxes = pingit.get_classification(img)[0]   #normalized bounding boxes coordinates
                scores = pingit.get_classification(img)[1]  #confidence for every detection
                headCount = 0

                for i in range(0,len(scores[0])):
                    if(scores[0][i] > list_thresholds[j]):
                        headCount += 1
                        #cv2.rectangle(new_img,(int(boxes[0][i][1] * 640), int(boxes[0][i][0] * 480)), (int(boxes[0][i][3] * 640), int(boxes[0][i][2] * 480)),(0,0,255), 3)
                if(headCount == gtHeadCount):
                    list_total_pos[j] += 1
                else:
                    list_total_neg[j] += 1
                if(gtHeadCount > 2):
                    if(headCount > 2):
                        list_binary_pos[j] += 1
                    else:
                        list_binary_neg[j] += 1
                #print(images[k] + "-- DETECTED: " + str(headCount) + "/" + str(gtHeadCount))
                #cv2.imwrite(out_folder + images[k],new_img)
    for m in range(0,len(list_thresholds)):
        exact_accuracy = ((list_total_pos[m])/(list_total_pos[m] + list_total_neg[m]))*100      #exact accuracy: accuracy where the number of detections = number of ground truth heads
        binary_accuracy = ((list_binary_pos[m])/(list_binary_pos[m] + list_binary_neg[m]))*100  #binary accuraccy: accuracy where both the detected heads and ground truth heads are greater than 2
        list_results[m] = exact_accuracy
        print("Threshold: " + str(list_thresholds[m]) + " -- Exact accuracy: " + str(exact_accuracy) + " -- Binary accuracy: " + str(binary_accuracy))
    matplotlib.pyplot.plot(np.array(list_results), np.array(list_thresholds))

def annotate(): #function similar to evaluate() in terms of principle, differs in that it draws the detected boxes and the ground truth boxes instead of calculating accuracies
    with tf.device('/gpu:0'):
        pingit = HeadClassifier()
    depth_folder = "/Data2TB/correctly_registered/S12/test/depth/"
    out_folder = "/home/wassimea/Desktop/SMATS/images/8bit/test_out/"
    with open("/home/wassimea/Desktop/wzebb.json") as f:
        jsonprop = json.load(f)

    images = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
    for image in images:
        rgb_image = cv2.imread("/Data2TB/correctly_registered/S12/test/color/" + image.replace(".png",".jpg"))
        cvimage = cv2.imread(depth_folder + image, -1)
        count = 0
        for frame in jsonprop["proposals"]:
            if frame["file"] == image:
                for object in frame["objects"]:
                    x = object["x"]
                    y = object["y"]
                    z = object["z"]
                    cv2.circle(rgb_image,(x,y),3,(0,0,255),2)
                    headpoint = [y,x]
                    xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
                    #cv2.rectangle(rgb_image,(xmiw,ymiw),(xmaw,ymaw),(255,0,0),2)
                    roi = cvimage[ymiw:ymaw, xmiw:xmaw]
                    width,height = roi.shape[1], roi.shape[0]
                    if(width > 0 and height > 0):# and z <= 4500):
                        headpoint[0] = headpoint[0] - ymiw
                        headpoint[1] = headpoint[1] - xmiw
                        c1,c2,c3,mod = core_functions.get_channels(roi, headpoint)
                        #cv2.imwrite("/home/wassimea/Desktop/chl_work/outs/" + str(count)+ ".jpg",mod)
                        #all = pingit.get_classification(mod)
                        boxes = pingit.get_classification(mod)[0]
                        scores = pingit.get_classification(mod)[1]
                        classes = pingit.get_classification(mod)[2]
                        scores_sorted = sorted(scores,reverse = True)
                        print(str(scores_sorted[0][0]))
                        #scores = sorted(scores,reverse = True)
                        maximum = max(scores)
                        maxclass = max(classes)
                        #cv2.imshow("ayre",mod)
                        #cv2.waitKey()
                        localcount = 0
                        for i in range(0,len(scores[0])):
                            if(scores[0][i] >= 0.003):
                                localcount = localcount + 1
                                if(classes[0][i] == 1.0):
                                    xmin = xmiw + int(boxes[0][i][1] * width)
                                    ymin = xmaw + int(boxes[0][i][0] * height)
                                    xmax = xmaw + int(boxes[0][i][3] * width)
                                    ymax = ymaw + int(boxes[0][i][2] * height)
                                    #cv2.rectangle(rgb_image,xmin, ymin, xmax, ymaw + ymax),(0,255,0), 3)
                                    cv2.rectangle(rgb_image,(xmiw-1,ymiw-1),(xmaw+1,ymaw+1),(0,0,255),2)
                                #else:
                                    #cv2.rectangle(mod,(int(boxes[0][i][1] * width), int(boxes[0][i][0] * height)), (int(boxes[0][i][3] * width), int(boxes[0][i][2] * height)),(255,0,0), 3)
                        if(localcount > 1):
                            toll = 0
                        #cv2.imwrite("/home/wassimea/Desktop/chl_work/outs/" + image,mod)
                        #print("check")
                        #cv2.waitKey()
                    count = count + 1
                cv2.imwrite("/home/wassimea/Desktop/chl_work/outs/" + image.replace(".png","_") + str(count) + ".jpg",rgb_image)

    
    for k in range(0,len(images)):
        if("resized-1-rotated-0" in images[k]):
            print(images[k])
            img = np.array(Image.open(folder + images[k]))
            new_img = img
            for i in range(0,len(jsondata['frames'])):
                    if(jsondata['frames'][i]['file'] == images[k]):
                        for j in range(0,len(jsondata['frames'][i]['annotations'])):
                            if(jsondata['frames'][i]['annotations'][j]['label'] == 'Head'):
                                xmin = jsondata['frames'][i]['annotations'][j]['x']
                                ymin = jsondata['frames'][i]['annotations'][j]['y']
                                xmax = xmin + jsondata['frames'][i]['annotations'][j]['width']
                                ymax = ymin + jsondata['frames'][i]['annotations'][j]['height']
                                cv2.rectangle(new_img,(xmin,ymin),(xmax,ymax),(255,0,0),3)

            boxes = pingit.get_classification(img)[0]
            scores = pingit.get_classification(img)[1]

            for i in range(0,len(scores[0])):
                if(scores[0][i] > 0.15):
                    cv2.rectangle(new_img,(int(boxes[0][i][1] * 640), int(boxes[0][i][0] * 480)), (int(boxes[0][i][3] * 640), int(boxes[0][i][2] * 480)),(0,0,255), 3)
            cv2.imwrite(out_folder + images[k],new_img)

def main(_):
    #evaluate()
    annotate()
if __name__ == '__main__':
  tf.app.run()
