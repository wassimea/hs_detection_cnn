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
def automate():
    index = 5000
    while(index <= 200000):
        os.mkdir("/Data2TB/chl_data/frozen/" + str(index))
        args = "--input_type=image_tensor --pipeline_config_path=/home/wassimea/Desktop/api/models/research/object_detection/chl_files/ssd_mobilenet_v2_chl.config --trained_checkpoint_prefix=/Data2TB/chl_data/model_dir/model.ckpt-" + str(index) + " --output_directory=/Data2TB/chl_data/frozen/" + str(index)
        #os.system("/home/wassimea/Desktop/api/models/research/object_detection/export_inference_graph.py " + args)
        call(['python3', '/home/wassimea/Desktop/api/models/research/object_detection/export_inference_graph.py', args])
        index += 5000

def main(_):
    automate()
if __name__ == '__main__':
  tf.app.run()
