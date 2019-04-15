#coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
#sys.path.append("../prepare_data")
print (sys.path)
import read_tfrecord_v2 #import read_multi_tfrecords,read_single_tfrecord
#from MTCNN_config import config
#from mtcnn_model import P_Net
import random
import numpy.random as npr
import cv2
import singleton as singleton
#tf.enable_eager_execution()
import mtcnn_model
import os

def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    config = singleton.configuration._instance.config
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    #lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    #control learning rate
    #global_step=tf.train.get_global_step()
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    # collect trainable parameters
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return train_op, lr_op
'''
certain samples mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    fliplandmarkindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip landmark    
    for i in fliplandmarkindexes:
        landmark_ = landmark_batch[i].reshape((-1,2))
        landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
        landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
        landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch
'''

def train(net_factory, prefix, end_epoch,display=200, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix:
    :param end_epoch:16
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    mylist = [1]
    l = np.array(mylist)
    config = singleton.configuration._instance.config
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train_gpu
    num = config.num

    pos_dir = os.path.join(config.train_pos_record)
    neg_dir = os.path.join(config.train_neg_record)

    pos_dir_val = os.path.join(config.val_pos_record)
    neg_dir_val = os.path.join(config.val_neg_record)

    dataset_dirs = [pos_dir,neg_dir]   #array containing the directories of the different tfrecord files
    dataset_dirs_val = [pos_dir_val,neg_dir_val]
    pos_radio = config.pos_radio
    neg_radio = config.neg_radio
    
    pos_batch_size = int(np.ceil(config.BATCH_SIZE*pos_radio))  #specifying how many positives in the batch
    assert pos_batch_size != 0,"Batch Size Error "

    #part_batch_size = int(np.ceil(config.BATCH_SIZE*part_radio))
    #assert part_batch_size != 0,"Batch Size Error "
    #         
    neg_batch_size = int(np.floor(config.BATCH_SIZE*neg_radio))  #specifying how many negatives in the batch
    assert neg_batch_size != 0,"Batch Size Error "
    
    #landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))    #specifying how many landmarks in the batch
    #assert landmark_batch_size != 0,"Batch Size Error "

    batch_sizes = [pos_batch_size,neg_batch_size]   #array of the distribution of pos,neg, landmarks in the batch
    
    # batch_size number of images,labels,bbox,and landmarks from different record files (pos,neg,landmarks). Distribution of image weights is preserved
    image_batch, label_batch,filename_batch = read_tfrecord_v2.read_multi_tfrecords(dataset_dirs,batch_sizes)        
    
    image_batch_val, label_batch_val,filename_batch_val = read_tfrecord_v2.read_multi_tfrecords(dataset_dirs_val,[100,100])  

    radio_cls_loss = config.radio_cls_loss

    image_size = config.image_size
    
    #define placeholders
    input_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, config.input_channels], name='input_image')
    label = tf.placeholder(tf.float32, shape=[None,2], name='label')

    #class,regression
    # get initial losses
    #cls_loss_op,L2_loss_op,accuracy_op,cls_prob_op = net_factory(input_image, label,training=True)
    cls_loss_op,L2_loss_op,cls_prob_op = net_factory(input_image, label,training=True)
    #train,update learning rate(3 loss)
    train_op, lr_op = train_model(base_lr, radio_cls_loss*cls_loss_op + L2_loss_op, num)
    # init
    init = tf.global_variables_initializer()

    total_parameters = 0
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    
    with tf.Session(config=config1) as sess:
        sess.run(tf.global_variables_initializer())
        #save model
        #saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(max_to_keep=1000000)
        #sess.run(init)
        #visualize some variables
        tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
        #tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
        summary_op = tf.summary.merge_all()
        logs_dir = "/home/wassimea/Desktop/wassimea/work/train_models/mn/logs"
        if os.path.exists(logs_dir) == False:
            os.mkdir(logs_dir)
        writer = tf.summary.FileWriter(logs_dir,sess.graph)
        #begin 
        coord = tf.train.Coordinator()
        #begin enqueue thread
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0
        #total steps
        MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
        epoch = 0
        sess.graph.finalize()    
        try:
            for step in range(MAX_STEP):
                try:
                    i = i + 1
                    if coord.should_stop():
                        print("coord must stop")
                        break
                    image_batch_array, label_batch_array, filename_batch_array = sess.run([image_batch, label_batch, filename_batch])
                    p = np.random.permutation(len(filename_batch_array))
                    image_batch_array = image_batch_array[p]
                    label_batch_array = label_batch_array[p]
                    filename_batch_array = filename_batch_array[p]
                    #random flip
                    #image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
                    '''
                    print image_batch_array.shape
                    print label_batch_array.shape
                    print bbox_batch_array.shape
                    print landmark_batch_array.shape
                    print label_batch_array[0]
                    print bbox_batch_array[0]
                    print landmark_batch_array[0]
                    '''
                    _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array})
                    z = (step+1) % display
                    if (step+1) % display == 0:

                        image_batch_array_val, label_batch_array_val, filename_batch_array_val = sess.run([image_batch_val, label_batch_val, filename_batch_val])
                        #acc = accuracy(cls_pred, labels_batch)
                        #han
                        cls_loss,L2_loss,lr,cls_prob = sess.run([cls_loss_op, L2_loss_op,lr_op,cls_prob_op],
                                                                    feed_dict={input_image: image_batch_array_val, label: label_batch_array_val})                
                        print("%s : Step: %d, cls loss: %4f,L2 loss: %4f,lr:%f " % (
                        datetime.now(), step+1, cls_loss, L2_loss, lr))
                    #save every two epochs
                    if i * config.BATCH_SIZE > num: 
                        if(len(l) == 1):
                            l = filename_batch_array_val
                        else:
                            new = 0
                            for new_arr_val in filename_batch_array_val:
                                contained = False
                                for old_arr_val in l:
                                    if new_arr_val == old_arr_val:
                                        contained = True
                                if(contained == False):
                                    new += 1
                            print("New eval images: ", new)
                            x = filename_batch_array_val
                        total_pos = 0
                        total_neg = 0
                        for i in range(200):
                            #pred = cls_prob[i][1]
                            posval = label_batch_array_val[i][1]
                            negval = label_batch_array_val[i][0]
                            ind = np.argmax(cls_prob[i])
                            if(ind == 1 and posval == 1.0):
                                total_pos += 1
                            if(ind == 0 and negval == 1.0):
                                total_neg += 1
                        epoch = epoch + 1
                        i = 0
                        posacc = total_pos/100
                        negacc = total_neg/100

                        summpos = tf.Summary()
                        summpos.value.add(tag="posacc", simple_value = posacc)
                        writer.add_summary (summpos, global_step=epoch*2) #act as global_step

                        summneg = tf.Summary()
                        summneg.value.add(tag="negacc", simple_value = negacc)
                        writer.add_summary (summneg, global_step=epoch*2) #act as global_step
                        saver.save(sess, prefix, global_step=epoch*2)
                        #tf.train.write_graph(sess.graph.as_graph_def(), prefix, 'tensorflowModel.pb', False)
                        print("Finished epocch ------- Posacc: " + str(posacc) + "-----Negacc: " + str(negacc))
                    writer.add_summary(summary,global_step=step)
                except EnvironmentError as error:
                    print(error)
                    y = 1
        except tf.errors.OutOfRangeError:
            print("完成！！！")
        finally:
            coord.request_stop()
            print("Before writing")
            writer.close()
        coord.join(threads)
        sess.close()