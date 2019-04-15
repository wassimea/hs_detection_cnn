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

def create_tf_example():
  count = 0
  counter = 0
  writer = tf.python_io.TFRecordWriter("/Data2TB/correctly_registered/augmented/train.record")   #output file


  #with open(filename) as f:
  #  content = f.readlines()
  #content = [x.strip() for x in content]
  #new_img = PIL.Image.new("L", (480, 640))
  #new_img.putdata(content)
  
  #with tf.gfile.GFile(filename, 'rb') as fid:
  #  encoded_jpg = fid.read()
  with open("/Data2TB/correctly_registered/augmented/combined/combined.json") as f:
    jsondata = json.load(f)
  for i in range(0,len(jsondata['frames'])):        #looping through JSON objects

    height = jsondata['frames'][i]["height"] # Image height
    width = jsondata['frames'][i]["width"] # Image width
    #filename = "/Data2TB/correctly_registered/augmented/combined/" + example # Filename of the image. Empty if image is not from file
    #encoded_image_data = None # Encoded image bytes
    filename_only = jsondata['frames'][i]['file']
    print(str(i) + ": " + filename_only)
    filename = "/Data2TB/correctly_registered/augmented/combined/images/" + filename_only
    with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_jpg = fid.read()
    xmins = [] 
    xmaxs = [] 
    ymins = [] 
    ymaxs = [] 

    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for j in range(0,len(jsondata['frames'][i]['annotations'])):
        if(jsondata['frames'][i]['annotations'][j]['label'] == 'Head'):
          xmin = (jsondata['frames'][i]['annotations'][j]['x'])/width
          xmax = (jsondata['frames'][i]['annotations'][j]['x'] + jsondata['frames'][i]['annotations'][j]['width'])/width
          ymin = (jsondata['frames'][i]['annotations'][j]['y'])/height
          ymax = (jsondata['frames'][i]['annotations'][j]['y'] + jsondata['frames'][i]['annotations'][j]['height'])/height
          if xmin > 1:
            xmin = 1.0
          if xmax > 1:
            xmax = 1.0
          if ymin >1:
            ymin = 1.0
          if ymax > 1:
            ymax = 1.0
          if(xmin > 1 or xmax > 1 or ymin >1 or ymax > 1):
            print("UNNORMALIZED STUFF")
          xmins.append(xmin)  
          xmaxs.append(xmax)
          ymins.append(ymin)
          ymaxs.append(ymax)
          classes_text.append('head')
          classes.append(1)
        #elif(jsondata['frames'][i]['annotations'][j]['label'] == 'Right Shoulder' or jsondata['frames'][i]['annotations'][j]['label'] == 'Left Shoulder'):   
        #  xmin = (jsondata['frames'][i]['annotations'][j]['x'])
        #  ymin = (jsondata['frames'][i]['annotations'][j]['y'])
        #  if(xmin + 2 > width):
        #    xmin = width - 2
        #  if(ymin + 2 > height):
        #    ymin = height - 2
        #  xmax = xmin + 2
        #  ymax = ymin + 2
        #  xminf = xmin/width
        #  xmaxf = xmax/width
        #  yminf = ymin/height
        #  ymaxf = ymax/height
#
        #  if(xminf > 1 or xmaxf > 1 or yminf >1 or ymaxf > 1):
        #    print("UNNORMALIZED STUFF")
        #  xmins.append(xminf)  
        #  xmaxs.append(xmaxf)
        #  ymins.append(yminf)
        #  ymaxs.append(ymaxf)
        #  classes_text.append('shoulder')
        #  classes.append(2)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/filename': dataset_util.bytes_feature(str.encode(filename)),
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(_): 
  create_tf_example()
    #writer.write(tf_example.SerializeToString())
  #writer.close()


if __name__ == '__main__':
  tf.app.run()
