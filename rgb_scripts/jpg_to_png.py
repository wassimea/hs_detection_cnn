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
import functions.common_functions as common_functions
import pyximport; pyximport.install()
import mod
import time



def convert():
  folder = "/Data2TB/chl_data/mod/train/neg_png/"
  files = [f for f in listdir(folder) if isfile(join(folder, f))]
  for image in files:
    jpg = cv2.imread(folder + image)
    cv2.imwrite(folder + image.replace(".jpg",".png"),jpg)

def main(_):
  convert()

if __name__ == '__main__':
  tf.app.run()
