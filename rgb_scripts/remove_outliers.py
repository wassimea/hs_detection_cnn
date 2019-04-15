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

source_folder = config.neg_rgb_out
reference_folder = config.train_neg_reference_folder

def remove():
  source_images = [f for f in listdir(source_folder) if isfile(join(source_folder, f))]

  for source_image in source_images:
      if(not os.path.exists(reference_folder + source_image)):
          os.remove(source_folder + source_image)
  

def main(_):
  remove()
  
if __name__ == '__main__':
  tf.app.run()