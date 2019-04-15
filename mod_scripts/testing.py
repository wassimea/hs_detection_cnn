import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
from os import listdir
from os.path import isfile, join
import sys
#from train_models.mtcnn_model import O_Net


def test():
    rgb_image = cv2.imread("/Data2TB/chl_data/rgb/train/pos_png/video_fifth_2018-04-23_CAM1_1524499032737_id_34_index_0.png")
    d_image = cv2.imread("/Data2TB/chl_data/depth_normalized/train/pos_png/video_fifth_2018-04-23_CAM1_1524499032737_id_34_index_0.png",cv2.IMREAD_GRAYSCALE)
    d_image = d_image.reshape((83,83,1))
    #cv2.normalize(d_image,  d_image, 0, 255, cv2.NORM_MINMAX)
    rgbd = np.concatenate((rgb_image, d_image), axis=2)
    s = rgbd.shape
    rgbd = cv2.resize(rgbd,(48,48))
    b,g,r,d = cv2.split(rgbd)
    rgb = cv2.merge((b,g,r))
    cv2.imshow("rgb",rgb)
    cv2.imshow("depth",d)
    cv2.waitKey()
    sd = d.shape
    x = 1

def main(_):
    test()
if __name__ == '__main__':
  tf.app.run()
