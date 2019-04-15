from skimage.io import imread
from skimage.io import imsave
from skimage.filters import rank
from skimage.morphology import rectangle
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
import random

chl_results_color_folder = "/Data2TB/correctly_registered/S12/S1/chl_results_color/"
color_folder = "/Data2TB/correctly_registered/S12/S1/color/"
depth_folder = "/Data2TB/correctly_registered/S12/S1/depth/"
depth_paletted_folder = "/Data2TB/correctly_registered/S12/S1/depth_paletted/"
overlayed_folder = "/Data2TB/correctly_registered/S12/S1/overlayed/"

train_out_folder = "/Data2TB/correctly_registered/S12/train/"
test_out_folder = "/Data2TB/correctly_registered/S12/test/"

png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
def split():
    count = 0
    while count != 161:
        rand = random.randint(0,len(png_files) - 1)
        filename = png_files[rand]
        filename_jpg = png_files[rand].replace("png","jpg")
        if os.path.isfile(depth_folder + filename):
            os.rename(chl_results_color_folder + filename_jpg, test_out_folder + "chl_results_color/" + filename_jpg )
            os.rename(color_folder + filename_jpg, test_out_folder + "color/" + filename_jpg )
            os.rename(depth_folder + filename,test_out_folder + "depth/" + filename)
            os.rename(depth_paletted_folder + filename_jpg, test_out_folder + "depth_paletted/" + filename_jpg )
            os.rename(overlayed_folder + filename_jpg, test_out_folder + "overlayed/" + filename_jpg )
            count = count + 1

def main(_): 
  print('Argument List:', str(sys.argv))
  split()

if __name__ == '__main__':
  tf.app.run()
