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

depth_folder = "/Data2TB/FastFood/S1/depth/"
png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]

def fix():
  #png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
  #Parallel(n_jobs=4)(delayed(processInput)(i) for i in range (0, len(png_files)))
  for i in range (0, len(png_files)):
      depth = imread(depth_folder + "/" + png_files[i])
      depth = rank.mean(depth, rectangle(3,3), mask=depth!=0)
      imsave("/Data2TB/FastFood/S1/depth_fixed/" + png_files[i],depth)
      #dcv = cv2.imread("/Data2TB/FastFood/S1/depth_fixed/" + png_files[i], -1)
      dcv = (depth/256).astype('uint8')
      #dcv = np.zeros((480,640,3), np.uint8)
      #dcv = cv2.cvtColor(depth,cv2.COLOR_GRAY2RGB,)
      cv2.imwrite("/Data2TB/FastFood/S1/depth_paletted_fixed/" + png_files[i],dcv)
      #cv2.imshow('zabri',dcv)
      #cv2.waitKey(1)

def processInput(i):
  file = png_files[i]
  depth = imread(depth_folder + file)
  depth = rank.mean(depth, rectangle(3,3), mask=depth!=0)
  imsave("/Data2TB/FastFood/S1/depth_fixed/" + file,depth)
  dcv = cv2.imread(depth_folder + file, -1)
  dcv = cv2.cvtColor(dcv,cv2.COLOR_GRAY2BGR)
  cv2.imshow('zabri',dcv)
  #depth = cv2.cvtColor(depth,depth,cv2.COLOR_GRAY2RGB, 0)
  #cv2.imshow('zabri',depth)
  cv2.waitKey(1)
def main(_): 
  print('Argument List:', str(sys.argv))
  fix()

if __name__ == '__main__':
  tf.app.run()
