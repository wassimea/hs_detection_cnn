#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7
import sys
#sys.path.append("../train_models/")
x = sys.path
sys.path.append("/home/wassimea/Desktop/wassimea/work/train_models")
import singleton as singleton
from mobilenet import mobilenet_v2
from mobilenet_git import mobilenet_v2 as mobilenet_git
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#define prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg
def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):
    #label = label[:, 0]
    #cls_prob = cls_prob[:, 0]
    #label = tf.reshape(label,[-1])
    #cls_prob = tf.reshape(cls_prob, [-1])
    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory

    #pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    #row = [0,2,4.....]
    #row = tf.range(num_row)*2
    #indices_ = row + label_int
    #label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(cls_prob+1e-10)
    zeros = tf.zeros_like(cls_prob, dtype=tf.float32)
    ones = tf.ones_like(cls_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros,zeros,ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds[0])

    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #FILTER OUT PART AND LANDMARK DATA
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


#def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
#    sigma = tf.constant(1.0)
#    threshold = 1.0/(sigma**2)
#    zeros_index = tf.zeros_like(label, dtype=tf.float32)
#    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
#    abs_error = tf.abs(bbox_pred-bbox_target)
#    loss_smaller = 0.5*((abs_error*sigma)**2)
#    loss_larger = abs_error-0.5/(sigma**2)
#    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
#    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
#    smooth_loss = smooth_loss*valid_inds
#    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
#    smooth_loss_picked = tf.gather(smooth_loss, k_index)
#    return tf.reduce_mean(smooth_loss_picked)
#def bbox_ohem_orginal(bbox_pred,bbox_target,label):
#    zeros_index = tf.zeros_like(label, dtype=tf.float32)
#    #pay attention :there is a bug!!!!
#    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
#    #(batch,)
#    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
#    #keep_num scalar
#    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
#    #keep valid index square_error
#    square_error = square_error*valid_inds
#    _, k_index = tf.nn.top_k(square_error, k=keep_num)
#    square_error = tf.gather(square_error, k_index)
#    return tf.reduce_mean(square_error)
#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred,bbox_target,label):
    pos_index = tf.constant([0])
    label = label[:, 1]
    zeros_index = tf.zeros_like(label, dtype=tf.float32)    #create a tensor like the label tensor, with all values set to 0
    ones_index = tf.ones_like(label,dtype=tf.float32)   #create a tensor like the label tensor, with all values set to 1
    valid_inds = tf.where(tf.equal(label, 1),ones_index,zeros_index)    #valid inds is an array of size batch_size, for each bbox tensor, if the corresponding label is 1, valid inds is 1. It is 0 otherwise
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target) #calculates the square of the differences between bbox_pred and bbox_target
    square_error = tf.reduce_sum(square_error,axis=1)   #sum of square errors on columns
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)   #number of positive samples that we want to evaluate the bbox loss on
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)   #cast to int32.
    #keep valid index square_error
    square_error = square_error*valid_inds  #multiplay square error by the number of positive samples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)  #k_index is the indices of the highest (keep_num) square errors. This way, we only get the loss of the positive samples in the batch  
    square_error = tf.gather(square_error, k_index) #get the relevant square errors (ones relating to positive samples)
    return tf.reduce_mean(square_error) #sum of relevant samples square error

def landmark_ohem(landmark_pred,landmark_target,label):
    pos_index = tf.constant([0])
    label = label[:, 1]
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)  #create a tensor like the label tensor, with all values set to 1
    zeros = tf.zeros_like(label,dtype=tf.float32)   #create a tensor like the label tensor, with all values set to 0
    valid_inds = tf.where(tf.equal(label,1),ones,zeros)     #valid inds is an array of size batch_size, for each landmark tensor, if the corresponding label is 1, valid inds is 1. It is 0 otherwise
    square_error = tf.square(landmark_pred-landmark_target) #calculates the square of the differences between landmark_pred and landmark_target
    square_error = tf.reduce_sum(square_error,axis=1)   #sum of square errors on columns
    num_valid = tf.reduce_sum(valid_inds)   #number of positive samples that we want to evaluate the landmark loss on
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)   #cast to int32.
    square_error = square_error*valid_inds   #multiplay square error by the number of positive samples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)  #k_index is the indices of the highest (keep_num) square errors. This way, we only get the loss of the positive samples in the batch  
    square_error = tf.gather(square_error, k_index) #get the relevant square errors (ones relating to positive samples)
    return tf.reduce_mean(square_error) #sum of relevant samples square error
    
def cal_accuracy(cls_prob,label):
    label = label[:, 0]
    pred = tf.argmax(cls_prob,axis=1)   #pred is the predicted class for every image in the batch
    label_int = tf.cast(label,tf.int64) #label_int is pred casted to an integer tensor
    cond = tf.where(tf.greater_equal(label_int,0))  #get the indices of positive samples in the batch
    print("=====")
    print(cond.get_shape())
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op

def get_bb_loss(bbox_pred,bbox_target,label):
    label_temp = label[:,0]
    zero = tf.constant(1, dtype=tf.float32)
    where = tf.not_equal(label_temp, zero)
    indices = tf.where(where)
    valid_targets = tf.gather(bbox_target, indices)
    valid_predictions = tf.gather(bbox_pred, indices)
    mse = tf.reduce_mean(tf.losses.mean_squared_error(valid_targets,valid_predictions))
    return mse

def get_landmark_loss(landmark_pred,landmark_target,label):
    label_temp = label[:,0]
    zero = tf.constant(1, dtype=tf.float32)
    where = tf.not_equal(label_temp, zero)
    indices = tf.where(where)
    valid_targets = tf.gather(landmark_target, indices)
    valid_predictions = tf.gather(landmark_pred, indices)
    mse = tf.reduce_mean(tf.losses.mean_squared_error(valid_targets,valid_predictions))
    return mse
#construct Pnet
#label:batch


def onet_cnn4(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):

        #model structure
        #
        #

        #print (inputs.get_shape())
        
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print("Conv 1: ", net.get_shape())

        #net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3,3], stride=2, scope="conv2")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print("Pool 1: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print("Conv 2: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv4")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print("Pool 2: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print("Conv 3: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv4")
        #print("Conv 4: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv6")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print("Pool 3: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print("Conv 4: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],padding="SAME",stride=1,scope="conv5")
        #print("Conv 5: ", net.get_shape())

        fc_flatten = slim.flatten(net)
        #print(fc_flatten.get_shape())

        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=4,scope="landmark_fc",activation_fn=None)
        #print(landmark_pred.get_shape())


#train
        if training:
            config = singleton.configuration._instance.config
            #cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob))
            #cls_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(label,cls_prob))
            #cls_loss = cls_ohem(cls_prob,label)
            cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob,from_logits=False))
            #bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            if config.bbox_loss == "mse":
                bbox_loss = get_bb_loss(bbox_pred,bbox_target,label)
            else:
                bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)

            if config.landmark_loss == "mse":
                landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            else:
                landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #bbox_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_target,bbox_pred))
            accuracy = cal_accuracy(cls_prob,label)
            #landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #landmark_loss = tf.reduce_mean(tf.losses.mean_squared_error(landmark_target,landmark_pred))
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy,cls_prob

        else:
            return cls_prob,bbox_pred,landmark_pred


def onet_cnn5(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):

        #model structure
        #
        #

        #print (inputs.get_shape())
        
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print("Conv 1: ", net.get_shape())

        #net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3,3], stride=2, scope="conv2")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print("Pool 1: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print("Conv 2: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv4")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print("Pool 2: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print("Conv 3: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv4")
        #print("Conv 4: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv6")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print("Pool 3: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print("Conv 4: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],padding="SAME",stride=1,scope="conv5")
        print("Conv 5: ", net.get_shape())

        fc_flatten = slim.flatten(net)
        #print(fc_flatten.get_shape())

        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=4,scope="landmark_fc",activation_fn=None)
        #print(landmark_pred.get_shape())


#train
        if training:
            config = singleton.configuration._instance.config
            #cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob))
            #cls_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(label,cls_prob))
            #cls_loss = cls_ohem(cls_prob,label)
            cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob,from_logits=False))
            #bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            if config.bbox_loss == "mse":
                bbox_loss = get_bb_loss(bbox_pred,bbox_target,label)
            else:
                bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
                
            if config.landmark_loss == "mse":
                landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            else:
                landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #bbox_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_target,bbox_pred))
            accuracy = cal_accuracy(cls_prob,label)
            #landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #landmark_loss = tf.reduce_mean(tf.losses.mean_squared_error(landmark_target,landmark_pred))
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy,cls_prob

        else:
            return cls_prob,bbox_pred,landmark_pred


def onet_cnn6(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),                        
                        padding='valid'):

        #model structure
        #
        #

        #print (inputs.get_shape())
        
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print("Conv 1: ", net.get_shape())

        #net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3,3], stride=2, scope="conv2")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print("Pool 1: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print("Conv 2: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv4")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print("Pool 2: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print("Conv 3: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],padding="SAME",stride=1,scope="conv4")
        print("Conv 4: ", net.get_shape())

        #net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=2,scope="conv6")
        #print(net.get_shape())

        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print("Pool 3: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv5")
        print("Conv 5: ", net.get_shape())

        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],padding="SAME",stride=1,scope="conv6")
        print("Conv 6: ", net.get_shape())

        fc_flatten = slim.flatten(net)
        #print(fc_flatten.get_shape())

        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #print(fc1.get_shape())
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #print(cls_prob.get_shape())
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #print(bbox_pred.get_shape())
        #batch*10
        landmark_pred = slim.fully_connected(fc1,num_outputs=4,scope="landmark_fc",activation_fn=None)
        #print(landmark_pred.get_shape())


#train
        if training:
            config = singleton.configuration._instance.config
            #cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob))
            #cls_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(label,cls_prob))
            #cls_loss = cls_ohem(cls_prob,label)
            cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob,from_logits=False))
            #bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            if config.bbox_loss == "mse":
                bbox_loss = get_bb_loss(bbox_pred,bbox_target,label)
            else:
                bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
                
            if config.landmark_loss == "mse":
                landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            else:
                landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #bbox_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_target,bbox_pred))
            accuracy = cal_accuracy(cls_prob,label)
            #landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
            #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            #landmark_loss = tf.reduce_mean(tf.losses.mean_squared_error(landmark_target,landmark_pred))
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy,cls_prob

        else:
            return cls_prob,bbox_pred,landmark_pred
            
        
def mobilenet(inputs,label=None,training=True):
    #cls_prob = tf.nn.softmax(logits,axis=None,name=None,dim=None)
    print("test")
    if training:
        config = singleton.configuration._instance.config
        #with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        #with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        #with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
        #    cls_prob, endpoints = mobilenet_v2.mobilenet(inputs)

        cls_prob, pred = mobilenet_git.mobilenetv2(inputs, 2, True)

        cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob,from_logits=True))
        #bbox_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_target,bbox_pred))
        #accuracy = cal_accuracy(cls_prob,label)
        #landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
        #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
        #landmark_loss = tf.reduce_mean(tf.losses.mean_squared_error(landmark_target,landmark_pred))
        #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
        L2_loss = tf.add_n(slim.losses.get_regularization_losses())
        return cls_loss,L2_loss,cls_prob

    else:
        #with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        #with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        cls_prob, pred = mobilenet_git.mobilenetv2(inputs, 2, False)
        cls_prob = tf.nn.softmax(cls_prob,axis=None,name=None,dim=None)
        #cls_prob, endpoints = mobilenet_v2.mobilenet(inputs)
        #cls_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label,cls_prob,from_logits=True))
        #bbox_loss = tf.reduce_mean(tf.losses.mean_squared_error(bbox_target,bbox_pred))
        #accuracy = cal_accuracy(cls_prob,label)
        #landmark_loss = get_landmark_loss(landmark_pred, landmark_target,label)
        #landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
        #landmark_loss = tf.reduce_mean(tf.losses.mean_squared_error(landmark_target,landmark_pred))
        #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
        #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
        return pred