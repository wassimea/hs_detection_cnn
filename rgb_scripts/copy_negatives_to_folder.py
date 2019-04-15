import tensorflow as tf
import sys
import os
import json
import PIL
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import functions.core_functions as core_functions
import shutil

src_folder = "/Data2TB/sample/"
ref_folder = "/Data2TB/correctly_registered/S12/train/negatives/overlayed/"
dest_folder = "/Data2TB/correctly_registered/S12/train/negatives/color/"

#convert(): function that gets 16 bit single channel png images in 'png_folder', binary images from 'binary_folder', and creates an image of 2 channels, 16 bits each from the two images 
def copy():
    rgb_files = [f for f in listdir(ref_folder) if isfile(join(ref_folder, f))]
    for i in range(0,len(rgb_files)):
        if(os.path.exists(src_folder + rgb_files[i])):
            shutil.copy(src_folder + rgb_files[i], dest_folder + rgb_files[i])
            print("ok")

def main(_): 
  print('Argument List:', str(sys.argv))
  copy()
if __name__ == '__main__':
  tf.app.run()
