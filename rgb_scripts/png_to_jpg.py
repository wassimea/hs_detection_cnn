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

png_folder = "/Data2TB/correctly_registered/S12/test/aggregated/0.8/"
png_files = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]

def fix():
  #png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
  #Parallel(n_jobs=4)(delayed(processInput)(i) for i in range (0, len(png_files)))
  for i in range (0, len(png_files)):
    image = cv2.imread(png_folder + png_files[i])
    cv2.imwrite(png_folder + png_files[i].replace(".png",".jpg"), image)
    #os.rename(png_folder + png_files[i], png_folder + png_files[i].replace(".jpg",".png"))
def main(_): 
  print('Argument List:', str(sys.argv))
  fix()

if __name__ == '__main__':
  tf.app.run()
