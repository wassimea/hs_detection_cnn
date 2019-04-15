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
from skimage.filters import rank
from skimage.morphology import rectangle
import functions.core_functions as core_functions
import time
import chl

def execute():
    times = []
    depth_folder = "/Data2TB/correctly_registered/S12/train/depth/"
    display_folder = "/Data2TB/correctly_registered/S12/train/color/"
    png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
    #png_image = cv2.imread("/Data2TB/correctly_registered/S12/train/depth/video_fifth_2018-04-23_CAM1_1524499198772.png",-1)
    #display_image = cv2.imread("/Data2TB/correctly_registered/S12/train/color/video_fifth_2018-04-23_CAM1_1524499198772.jpg")
    #ref_image = cv2.imread("/Data2TB/correctly_registered/S12/train/chl_results_color/video_fifth_2018-04-23_CAM1_1524499198772.jpg")
    for image in png_files:
        #image = "video_fifth_2018-04-23_CAM1_1524501445409.png"
        #print(image)
        png_image = cv2.imread(depth_folder + image, -1)
        #display_image = cv2.imread(display_folder + image.replace(".png",".jpg"), -1)
        startchl = time.clock()
        chl_results = chl.findz(png_image)
        #for prop in chl_results:
        #    #cv2.circle(display_image,(prop[1], prop[2]), 3, (0,0,255), -1)
        #    cv2.rectangle(display_image,(prop[0], prop[1]), (prop[2], prop[3]),(0,0,255), 1)
        times.append(time.clock() - startchl)
        #print("image end")
        #cv2.imshow("chl",display_image)
        #cv2.waitKey()

    #png_image = png_image.astype(np.int16)

    #for i in range(0,100):
    #    startchl = time.clock()
    #    chl_results = chl.findz(png_image)
    #    #timechl = time.clock() - startchl
    #    for prop in chl_results:
    #        cv2.circle(display_image,(prop[1], prop[2]), 3, (0,0,255), -1)
    #    cv2.imshow("chl",display_image)
    #    #cv2.imshow("ref", ref_image)
    #    cv2.waitKey()
    #    times.append(time.clock() - startchl)
    
    #print(times)
    print(sum(times)/len(png_files))

    #for prop in chl_results:
    #    cv2.circle(display_image,(prop[1], prop[2]), 3, (0,0,255), -1)
    #cv2.imshow("chl",display_image)
    #cv2.imshow("ref", ref_image)
    #cv2.waitKey()


def main(_):
    execute()

if __name__ == '__main__':
  tf.app.run()