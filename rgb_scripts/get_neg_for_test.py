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
import random
from shutil import copyfile

neg_folder = "/Data2TB/chl_data/rgb/test/neg/"
neg_for_test_folder = "/Data2TB/chl_data/rgb/test/neg_for_test/"

def process():
    negs = [f for f in listdir(neg_folder) if isfile(join(neg_folder, f))]
    count = 0
    while count < 170:
        index = random.randint(0,len(negs))
        if not os.path.isfile(neg_for_test_folder + negs[index]):
            copyfile(neg_folder + negs[index],neg_for_test_folder + negs[index])
            count = count + 1
            print(count)


def main(_):  
    process()
if __name__ == '__main__':
  tf.app.run()
