#coding:utf-8

import tensorflow as tf
import numpy as np
import cv2
import os
from MTCNN_config import config
import train_models.singleton as singleton
#just for RNet and ONet, since I change the method of making tfrecord
#as for PNet

def read_single_tfrecord_3c(tfrecord_file, batch_size):

    #tfrecord reading
    #
    #
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([2], tf.float32),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([4],tf.float32)
        }
    )
    image_size = 48
    #image = tf.decode_raw(image_features['image/encoded'], tf.uint8)

    #decode and normalize image
    #
    #

    image = tf.image.decode_image(image_features['image/encoded'], channels=3)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = (tf.cast(image, tf.float32)-127.5) / 128

    filename = tf.cast(image_features['image/filename'],tf.string)

    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['image/label'], tf.float32)  #get image label
    roi = tf.cast(image_features['image/roi'],tf.float32)   #get image roi
    landmark = tf.cast(image_features['image/landmark'],tf.float32) #get image landmark
    image, label,roi,landmark,filename = tf.train.batch(
        [image, label,roi,landmark,filename],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size,2])
    roi = tf.reshape(roi,[batch_size,4])
    landmark = tf.reshape(landmark,[batch_size,4])
    #print("========================================================================")
    #print ("LANDMARK SHAPE: ",landmark.get_shape())
    return image, label, roi,landmark, filename

def read_single_tfrecord_4c(tfrecord_file, batch_size):

    #tfrecord reading
    #
    #
    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/encoded_d': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([2], tf.float32),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([4],tf.float32)
        }
    )
    image_size = 48
    #image = tf.decode_raw(image_features['image/encoded'], tf.uint8)

    #decode and normalize image
    #
    #

    image = tf.image.decode_image(image_features['image/encoded'], channels=3)
    image_d = tf.image.decode_image(image_features['image/encoded_d'], channels=1)
    image = tf.concat([image,image_d],axis=2)
    image = tf.reshape(image, [image_size, image_size, 4])
    image = (tf.cast(image, tf.float32)-127.5) / 128

    filename = tf.cast(image_features['image/filename'],tf.string)

    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['image/label'], tf.float32)  #get image label
    roi = tf.cast(image_features['image/roi'],tf.float32)   #get image roi
    landmark = tf.cast(image_features['image/landmark'],tf.float32) #get image landmark
    image, label,roi,landmark,filename = tf.train.batch(
        [image, label,roi,landmark,filename],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size,2])
    roi = tf.reshape(roi,[batch_size,4])
    landmark = tf.reshape(landmark,[batch_size,4])
    #print("========================================================================")
    #print ("LANDMARK SHAPE: ",landmark.get_shape())
    return image, label, roi,landmark, filename

def read_multi_tfrecords(tfrecord_files, batch_sizes):
    config = singleton.configuration._instance.config
    pos_dir,neg_dir, = tfrecord_files   #getting the tfrecord files for positive,negative, and landmark from tfrecord_files array

    pos_batch_size,neg_batch_size, = batch_sizes #getting the batch sizes or the distribution of positive,negative, and landmark in the batch being generated

    #assert net=='RNet' or net=='ONet', "only for RNet and ONet"
    if(config.input_channels == 4):
        pos_image,pos_label,pos_roi,pos_landmark, pos_filename = read_single_tfrecord_4c(pos_dir, pos_batch_size)   #read pos_batch_size number of positive samples with their labels,rois, and landmarks
        neg_image,neg_label,neg_roi,neg_landmark,neg_filename = read_single_tfrecord_4c(neg_dir, neg_batch_size)   #read neg_batch_size number of positive samples with their labels,rois, and landmarks
    else:
        pos_image,pos_label,pos_roi,pos_landmark, pos_filename = read_single_tfrecord_3c(pos_dir, pos_batch_size)   #read pos_batch_size number of positive samples with their labels,rois, and landmarks
        neg_image,neg_label,neg_roi,neg_landmark,neg_filename = read_single_tfrecord_3c(neg_dir, neg_batch_size)   #read neg_batch_size number of positive samples with their labels,rois, and landmarks
    
    print (pos_image.get_shape())
    print (neg_image.get_shape())

    #landmark_image,landmark_label,landmark_roi,landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, net) #read landmark_batch_size number of positive samples with their labels,rois, and landmarks
    #print (landmark_image.get_shape())
    
    images = tf.concat([pos_image,neg_image], 0, name="concat/image")    #concatenate image tensors from different records to 1D
    print (images.get_shape())

    labels = tf.concat([pos_label,neg_label],0,name="concat/label")  #concatenate label tensors from different records to 1D
    print (labels.get_shape())

    rois = tf.concat([pos_roi,neg_roi],0,name="concat/roi")    #concatenate roi tensors from different records to 1D
    print (rois.get_shape())

    landmarks = tf.concat([pos_landmark,neg_landmark],0,name="concat/landmark")   #concatenate landmark tensors from different records to 1D
    print (landmarks.get_shape())

    filenames = tf.concat([pos_filename,neg_filename],0,name="concat/landmark")
    print (filenames.get_shape())

    return images,labels,rois,landmarks,filenames #return a batch with size batch_size of images, labels, rois, and landmarks from different samples of pos,neg and landmarks
    