import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("../")
from train_models.mtcnn_model import O_Net
from os import listdir
from os.path import isfile, join
import json

class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self,index):#, data_size, batch_size, model_path):
        net_factory = O_Net
        model_path = '/home/wassimea/Desktop/wassimea/work/train_models/model_path/'
        batch_size = 1
        data_size = 48
        graph = tf.Graph()
        #tf.reset_default_graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
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
            saver.restore(self.sess, "/home/wassimea/Desktop/wassimea/work/train_models/model_path/-" + str(index))                        

        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self,image):
        # access data
        # databatch: N x 3 x data_size x data_size
        #databatch = 1*3*48*48
        cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: image})
        return cls_prob, bbox_pred, landmark_pred
        
def predictim(index):
    data ={}
    data['results'] = []
    pos_folder = "/Data2TB/chl_data/rgb/test/pos_png/"
    neg_folder = "/Data2TB/chl_data/rgb/test/neg_png/"
    threshold = 0.15
    posacc = 0
    pingit = Detector(index)
    pos_images = [f for f in listdir(pos_folder) if isfile(join(pos_folder, f))]
    neg_images = [f for f in listdir(neg_folder) if isfile(join(neg_folder, f))]
    for i in range(0,len(pos_images)):

        #image_buffer = cv2.imread(pos_folder + "video_fifth_2018-04-23_CAM1_1524501165653_id_665.png")
        #image_buffer = cv2.imencode('.png', image_buffer)[1].tostring()
        #with tf.gfile.GFile(pos_folder + pos_images[i], 'rb') as fid:
        #    encoded_jpg = fid.read()
        ##wassim = cv2.imdecode(image_buffer,cv2.IMREAD_ANYCOLOR)
        #image = tf.image.decode_image(encoded_jpg, channels=3)
        ##image = cv2.imread(pos_folder + "video_fifth_2018-04-23_CAM1_1524501165653_id_665.png")
        #image = tf.reshape(image, [1, 48, 48, 3])
        #image = (tf.cast(image, tf.float32)-127.5) / 128
        #image = read_single_tfrecord("/Data2TB/chl_data/rgb/val/records/pos.record",1)

        image2 = cv2.imread(pos_folder + pos_images[i])
        image2 = cv2.resize(image2,(48,48))
        roi = image2.copy()
        roi = roi.reshape(1, 48,48,3)
        roi = (roi.astype('float32') - 127.5)/128
        roi = roi[...,::-1]

        cls_scores, reg,landmark = pingit.predict(roi)

        #roi = cv2.resize(image,(48,48))
        #roi = tf.reshape(roi, [1, 48, 48, 3])
        #roi = (tf.cast(roi, tf.float32)-127.5) / 128
        roitmp = roi.copy()
        roitmp = np.reshape(roitmp, [1, 48, 48, 3])
        roitmp = (roitmp - 127.5)/128

        #cls_scores, reg,landmark = pingit.predict(roitmp)
        #keep_inds = np.where(cls_scores > threshold)
        classi = cls_scores[0]
        bb = reg[0]
        lm = landmark[0]

        xmin = int(bb[0] * 48)
        ymin = int(bb[1] * 48)
        xmax = int(bb[2] * 48)
        ymax = int(bb[3] * 48)

        xls = int(lm[0] * 48)
        yls = int(lm[1] * 48)

        xrs = int(lm[2] * 48)
        yrs = int(lm[3] * 48)

        #indexd = np.argmax(classi)#classi.index(max(classi))
        #if indexd == 1:
        ind = np.argmax(cls_scores[0])
        if(ind == 1):
        #if(classi[1] > classi[0] and (classi[1] - classi[0] >= threshold)):
            posacc += 1
            #cv2.rectangle(test, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            #cv2.circle(test, (yls,xls), 2, (0,255,0), thickness=1, lineType=8, shift=0)
            #cv2.circle(test, (yrs,xrs), 2, (0,255,0), thickness=1, lineType=8, shift=0)
#
            #cv2.imshow("test",test)
            #cv2.waitKey()
        #x = 1
    posacc = posacc/len(pos_images)
    print("Positive: ", posacc)

    negacc = 0
    for i in range(0,len(neg_images)):

        with tf.gfile.GFile(neg_folder + neg_images[i], 'rb') as fid:
            encoded_jpg = fid.read()
        #wassim = cv2.imdecode(image_buffer,cv2.IMREAD_ANYCOLOR)
        #image = tf.image.decode_image(encoded_jpg, channels=3)
        ##image = cv2.imread(pos_folder + "video_fifth_2018-04-23_CAM1_1524501165653_id_665.png")
        #image = tf.reshape(image, [1, 48, 48, 3])
        #image = (tf.cast(image, tf.float32)-127.5) / 128

        #roi = cv2.resize(image,(48,48))
        #roi = tf.reshape(roi, [1, 48, 48, 3])
        #roi = (tf.cast(roi, tf.float32)-127.5) / 128

        #roi = np.reshape(roi, [1, 48, 48, 3])
        #roi = (roi - 127.5)/128

        image2 = cv2.imread(neg_folder + neg_images[i])
        image2 = cv2.resize(image2,(48,48))
        roi = image2.copy()
        roi = roi.reshape(1, 48,48,3)
        roi = (roi.astype('float32') - 127.5)/128
        roi = roi[...,::-1]

        cls_scores, reg,landmark = pingit.predict(roi)
        #keep_inds = np.where(cls_scores > threshold)
        classi = cls_scores[0]
        #bb = reg[0]
        #lm = landmark[0]
#
        #xmin = bb[0]
        #ymin = bb[1]
        #xmax = bb[2]
        #ymax = bb[3]
#
        #xls = lm[0]
        #yls = lm[1]
#
        #xrs = lm[2]
        #yrs = lm[3]

        #indexd = np.argmax(classi)#classi.index(max(classi))
        #if indexd == 0:
        ind = np.argmax(cls_scores[0])
        if(ind == 0):
            negacc += 1
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            #cv2.circle(image, (xls,yls), 2, (0,255,0), thickness=1, lineType=8, shift=0)
            #cv2.circle(image, (xrs,yrs), 2, (0,255,0), thickness=1, lineType=8, shift=0)

        #cv2.imshow("test",image)
        #cv2.waitKey()
        #x = 1
        #else:
        #    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
        #    cv2.imshow("test",image)
        #    cv2.waitKey()

    negacc = negacc/len(neg_images)
    print("Negative: ", negacc)

    data['results'].append({
                'at' : str(index),
                'recall' : posacc,
                'FP' : negacc
            })
    if(posacc > 0.8 and negacc > 0.98):
        with open('/home/wassimea/Desktop/wassimea/work/detection/results.json', 'a') as outfile:  
            json.dump(data, outfile, indent=4)
    #outfile = open('/home/wassimea/Desktop/wassimea/work/detection/results.json', 'a')
    #json.dump(data, outfile,indent=4)

def main(_):
    f= open("/home/wassimea/Desktop/wassimea/work/detection/results.json","w+")
    f.close()
    index = 2
    while index <= 1118:
        print("Index: ", index)
        predictim(index)
        index += 2
if __name__ == '__main__':
  tf.app.run()
