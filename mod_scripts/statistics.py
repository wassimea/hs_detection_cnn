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

#convert(): function that gets 16 bit single channel png images in 'png_folder', binary images from 'binary_folder', and creates an image of 2 channels, 16 bits each from the two images 
def generate():
  gtheads = 0
  propheads = 0
  image_count = 0
  propcountarr = []
  depth_folder = "/Data2TB/correctly_registered/S2/tester/"
  with open("/home/wassimea/Desktop/wzebb.json") as f:
      jsonprop = json.load(f)
  with open("/Data2TB/sample/annotations.json") as f:
      jsongt = json.load(f)
  png_files = [f for f in listdir(depth_folder) if isfile(join(depth_folder, f))]
  for i in range(0,len(png_files)):
    image_name = png_files[i]#jsonprop["proposals"][i]["file"]
    image_name_jpg = image_name.replace("png","jpg")
    if image_name_jpg in jsongt:
      image_count = image_count + 1
      paletted_image = cv2.imread("/Data2TB/correctly_registered/S2/tester/paletted/" + image_name_jpg)
      gtarray = []#np.zeros(shape = (0,4))
      for k in range (0, len(jsongt[image_name_jpg]["annotations"])):
        if(jsongt[image_name_jpg]["annotations"][k]["category"] == "Head"):
          xmingt = jsongt[image_name_jpg]["annotations"][k]["x"] + 5
          ymingt = jsongt[image_name_jpg]["annotations"][k]["y"]
          width = jsongt[image_name_jpg]["annotations"][k]["width"]
          height = jsongt[image_name_jpg]["annotations"][k]["height"]
          xmaxgt = xmingt + width
          ymaxgt = ymingt + height
          if(xmingt >= 0 and ymingt >= 0 and xmingt <= 1000 and ymingt <= 1000):
            gtheads = gtheads + 1
            gtarray.append([xmingt,ymingt, xmaxgt,ymaxgt])# = np.append(gtarray, [xmingt,ymingt, xmaxgt,ymaxgt])
            cv2.rectangle(paletted_image,(xmingt,ymingt),(xmaxgt ,ymaxgt),(0,0,255),3)
      #os.mkdir("/home/wassimea/Desktop/testing/" + str(i))
      proparray = []#np.zeros(shape = (0,4))
      for j in range(0, len(jsonprop["proposals"])):#[i]["objects"])):
        if (jsonprop["proposals"][j]["file"] == image_name):
          propcount = 0
          for k in range (0,len(jsonprop["proposals"][j]["objects"])):
            propcount = propcount + 1
            x = jsonprop["proposals"][j]["objects"][k]["x"]
            y = jsonprop["proposals"][j]["objects"][k]["y"]
            z = jsonprop["proposals"][j]["objects"][k]["z"]
            cv2.circle(paletted_image,(x,y), 2, (255,0,0), 3)
            xmiw,ymiw,xmaw,ymaw = core_functions.get_bounding_box_WASSIMEA(x,y,z)
            proparray.append([xmiw,ymiw,xmaw,ymaw])
            cv2.rectangle(paletted_image,(xmiw,ymiw),(xmaw ,ymaw),(0,255,0),3)
          propcountarr.append(propcount)
      cv2.imshow("AYRE",paletted_image)
      cv2.waitKey()
      #cv2.imwrite("/home/wassimea/Desktop/testing/" + image_name.replace("png","jpg"),paletted_image)
      for m in range (0, len(gtarray)):
        for n in range (0, len(proparray)):
          iou = core_functions.bb_intersection_over_union(gtarray[m],proparray[n])
          contained =  False#check_if_rectangle_contained(gtarray[m],proparray[n])
          if(iou > 0.5):
            propheads = propheads + 1
            contained = True
            break
        if(contained == False):
          print(image_name_jpg)
          #cv2.imshow("kissimmik", paletted_image)
          #cv2.waitKey()
  zabre = np.mean(propcountarr)
  pingit = 1
def main(_): 
  print('Argument List:', str(sys.argv))
  generate()

if __name__ == '__main__':
  tf.app.run()
