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
from automation_config import config
import functions.core_functions as core_functions
import functions.common_functions as common_functions
import shutil

source_folder = "/Data2TB/correctly_registered/S12/test/depth/"
destination_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"

def normalize():
  source_images = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]

  for source_image in source_images:
        depth_image = cv2.imread(source_folder + source_image)
        cv2.normalize(depth_image,  depth_image, 0, 255, cv2.NORM_MINMAX)
        b,g,r = cv2.split(depth_image)
        cv2.imwrite(destination_folder + source_image, b)
        #cv2.waitKey()
  

def main(_):
  normalize()
  
if __name__ == '__main__':
  tf.app.run()