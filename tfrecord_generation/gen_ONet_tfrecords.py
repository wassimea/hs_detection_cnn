import tensorflow as tf
import sys
import os
import json
import dataset_util
import PIL
import cv2

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example():
  count = 0
  counter = 0
  writer = tf.python_io.TFRecordWriter("/Data2TB/chl_data/rgb/train/augmented/1to1/train.record")   #output file


  #with open(filename) as f:
  #  content = f.readlines()
  #content = [x.strip() for x in content]
  #new_img = PIL.Image.new("L", (480, 640))
  #new_img.putdata(content)
  
  #with tf.gfile.GFile(filename, 'rb') as fid:
  #  encoded_jpg = fid.read()
  with open("/Data2TB/chl_data/rgb/train/augmented/1to1/train_pos_neg_48.json") as f:
    jsondata = json.load(f)
  for i in range(0,len(jsondata['frames'])):        #looping through JSON objects

    height = jsondata['frames'][i]["height"] # Image height
    width = jsondata['frames'][i]["width"] # Image width
    #filename = "/Data2TB/correctly_registered/augmented/combined/" + example # Filename of the image. Empty if image is not from file
    #encoded_image_data = None # Encoded image bytes
    filename_only = jsondata['frames'][i]['file']
    filename = "/Data2TB/chl_data/rgb/train/augmented/1to1/pos_neg_png_48/" + filename_only
    image_buffer = cv2.imread(filename)
    image_buffer = cv2.imencode('.png', image_buffer)[1].tostring()
    x = len(image_buffer)
    with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_jpg = fid.read()
    y = len(encoded_jpg)
    xminh = 0 
    xmaxh = 0 
    yminh = 0
    ymaxh = 0

    xls = 0
    yls = 0

    xrs = 0
    yrs = 0

    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for j in range(0,len(jsondata['frames'][i]['annotations'])):
        if(jsondata['frames'][i]['annotations'][j]['label'] == 'Head'):
          xminh = (jsondata['frames'][i]['annotations'][j]['x'])/48
          xmaxh = (jsondata['frames'][i]['annotations'][j]['x'] + jsondata['frames'][i]['annotations'][j]['width'])/48
          yminh = (jsondata['frames'][i]['annotations'][j]['y'])/48
          ymaxh = (jsondata['frames'][i]['annotations'][j]['y'] + jsondata['frames'][i]['annotations'][j]['height'])/48
          #classes_text.append('head')
          #classes.append(1)
        elif(jsondata['frames'][i]['annotations'][j]['label'] == 'Right Shoulder' or jsondata['frames'][i]['annotations'][j]['label'] == 'Left Shoulder'):   
          xmin = (jsondata['frames'][i]['annotations'][j]['x'])/48
          ymin = (jsondata['frames'][i]['annotations'][j]['y'])/48
          if(xmin < 0):
              xmin = 0
          if(ymin > height):
              ymin = 1
          if(xmin > width):
              xmin = 1
          
          if(jsondata['frames'][i]['annotations'][j]['label'] == 'Left Shoulder'):
              xls = xmin
              yls = ymin
          elif(jsondata['frames'][i]['annotations'][j]['label'] == 'Right Shoulder'):
              xrs = xmin
              yrs = ymin

    if(xminh != 0 or yminh != 0 or xmaxh != 0 or ymaxh != 0):
        label = 1
    else:
        label = 0
    print(str(i) + ": " + str(label))
    roi = [xminh,yminh,xmaxh,ymaxh]
    shoulders = [xls,yls,xrs,yrs]

    #if(label == 1):
    #    shoulders_temp = [0,0,0,0]
    #    tf_example = tf.train.Example(features=tf.train.Features(feature={
    #    'image/encoded': _bytes_feature(encoded_jpg),
    #    'image/label': _int64_feature(label),
    #    'image/roi': _float_feature(roi),
    #    'image/landmark': _float_feature(shoulders_temp)
    #    }))
    #    writer.write(tf_example.SerializeToString())
#
    #    roi_temp = [0,0,0,0]
    #    label = -2
    #    tf_example = tf.train.Example(features=tf.train.Features(feature={
    #    'image/encoded': _bytes_feature(encoded_jpg),
    #    'image/label': _int64_feature(label),
    #    'image/roi': _float_feature(roi_temp),
    #    'image/landmark': _float_feature(shoulders)
    #    }))
    #    writer.write(tf_example.SerializeToString())
    #else:
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/label': _int64_feature(label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(shoulders)
    }))
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(_): 
  create_tf_example()
    #writer.write(tf_example.SerializeToString())
  #writer.close()


if __name__ == '__main__':
  tf.app.run()