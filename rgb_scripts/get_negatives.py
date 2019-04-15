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
from automation_config import config
import mod 

rgb_folder = config.neg_rgb_in
out_folder = config.neg_rgb_out

#convert(): function that gets 16 bit single channel png images in 'png_folder', binary images from 'binary_folder', and creates an image of 2 channels, 16 bits each from the two images 
def generate():
  propcountarr = []
  with open(config.prop_json) as f:
      jsonprop = json.load(f)
  rgb_files = [f for f in listdir(rgb_folder) if isfile(join(rgb_folder, f))]
  for i in range(0,len(rgb_files)):
    #print(rgb_files[i])
    image_name = rgb_files[i]#jsonprop["proposals"][i]["file"]
    image_name_png = image_name.replace("jpg","png")
    count = 0
    for j in range(0, len(jsonprop)):#[i]["objects"])):
      if (jsonprop[j]["file"] == image_name_png):
        propcount = 0
        for k in range (0,len(jsonprop[j]["objects"])):
          propcount = propcount + 1
          x = jsonprop[j]["objects"][k]["x"]
          y = jsonprop[j]["objects"][k]["y"]
          z = jsonprop[j]["objects"][k]["z"]
          xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
          prop_object = [xmiw,ymiw,xmaw,ymaw,y,x]
          headpoint = [prop_object[4],prop_object[5]]
          image, headpoint_new = core_functions.get_neg_roi(rgb_folder + image_name, prop_object, headpoint)
          if config.mod == True:
            image = mod.get_channels(image,headpoint_new)
            cv2.normalize(image,  image, 0, 255, cv2.NORM_MINMAX)
          width,height = image.shape[1], image.shape[0]
          if(width > 30 and height > 30):
            #c1,c2,c3,mod = core_functions.get_channels(image, headpoint_new)
            cv2.imwrite(out_folder + image_name.replace(".jpg", "_id_") + str(count) + ".jpg", image)
            count = count + 1
  zabre = np.mean(propcountarr)
  pingit = 1
def main(_): 
  print('Argument List:', str(sys.argv))
  generate()
if __name__ == '__main__':
  tf.app.run()
