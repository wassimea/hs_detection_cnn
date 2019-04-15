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

png_folder = config.neg_rgb_out
camera_factor = 1
camera_cx = 325.5
camera_cy = 253.5
camera_fx = 518.0
camera_fy = 519.0
data = {}
def process_image(filename):
    image = cv2.imread(png_folder + filename)
    width, height = image.shape[1], image.shape[0]
    data['frames'].append({
                            'file' : filename,
                            'width' : width,
                            'height' : height,
                            'annotations' : 
                            [
                            ]
                        })

def main(_):  
    data['frames'] = []
    png_files = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
    for i in range(0,len(png_files)):
        process_image(png_files[i])
    with open(config.neg_json_out, 'w') as outfile:  
        json.dump(data, outfile, indent=4)
if __name__ == '__main__':
  tf.app.run()
