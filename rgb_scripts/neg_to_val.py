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

reference_folder = config.val_reference_folder
source_folder = config.val_source_folder
destination_folder = config.val_destination_folder

def move():
  pos_images = [f for f in listdir(reference_folder + "pos_png/") if isfile(join(reference_folder + "pos_png/", f))]
  neg_images = [f for f in listdir(reference_folder + "neg_png/") if isfile(join(reference_folder + "neg_png/", f))]

  for pos_image in pos_images:
    shutil.move(source_folder + "pos_png/" + pos_image, destination_folder + "pos_png/" + pos_image,)
  for neg_image in neg_images:
    shutil.move(source_folder + "neg_png/" + neg_image, destination_folder + "neg_png/" + neg_image,)
  

def main(_):
  move()
  
if __name__ == '__main__':
  tf.app.run()