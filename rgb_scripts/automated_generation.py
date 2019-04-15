import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
from subprocess import call
from automation_config import config

def automate():
    #os.mkdir("/Data2TB/chl_data/frozen/" + str(index))
    if not os.path.exists(config.pos_rgb_out):
        os.mkdir(config.pos_rgb_out)

    if not os.path.exists(config.neg_rgb_out):
        os.mkdir(config.neg_rgb_out)

    print("Extracting data...")
    call(['python3', '/home/wassimea/Desktop/chl_work/rgb_scripts/extract_data.py'])
    print("Success")

    #print("Generating negatives...")
    #call(['python3', '/home/wassimea/Desktop/chl_work/rgb_scripts/get_negatives.py'])
    #print("Success")
 
    print("Generating JSON for negatives...")
    call(['python3', '/home/wassimea/Desktop/chl_work/rgb_scripts/generate_json_negatives.py'])
    print("Success")
#
    print("Moving from neg to val...")
    call(['python3', '/home/wassimea/Desktop/chl_work/rgb_scripts/neg_to_val.py'])
    print("Success")

    print("Removing outliers...")
    call(['python3', '/home/wassimea/Desktop/chl_work/rgb_scripts/remove_outliers.py'])
    print("Success")

def main(_):
    automate()
if __name__ == '__main__':
  tf.app.run()
