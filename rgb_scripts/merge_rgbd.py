import tensorflow as tf
import numpy as np
import cv2
import sys
import os
from os import listdir
from os.path import isfile, join
import json


def generate():
    alpha = 0.9
    color_folder = "/Data2TB/correctly_registered/S12/test/color/"
    depth_folder = "/Data2TB/correctly_registered/S12/test/depth/"

    color_images = [f for f in listdir(color_folder) if isfile(join(color_folder, f))]

#while alpha > 0:
    beta = round(1 - alpha,1)
    out_folder = "/Data2TB/correctly_registered/S12/test/rgbd/" + str(beta)
    os.makedirs(out_folder)
    for image in color_images:
        rgb_image = cv2.imread(color_folder + image)
        d_image = cv2.imread(depth_folder + image.replace(".jpg",".png"),cv2.IMREAD_COLOR)
        cv2.normalize(d_image,  d_image, 0, 255, cv2.NORM_MINMAX)

        rgbd_image = cv2.addWeighted(rgb_image,alpha,d_image,beta,0)
        #cv2.imshow("im",rgbd_image)
        #cv2.waitKey()

        cv2.imwrite(out_folder + "/" + image.replace(".jpg",".png"),rgbd_image)
    #alpha = round(alpha - 0.1,1)





def main(_):
    generate()
if __name__ == '__main__':
  tf.app.run()
