#coding:utf-8
import mtcnn_model
#from MTCNN_config import config
#import mtcnn_model import O_Net
from train import train
import train_models.singleton as singleton
import sys
import os
def train_ONet(prefix, end_epoch, display, lr):
    """
    train ONet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    config = singleton.configuration._instance.config
    if(config.mtcnn_model == "onet_cnn4"):
        net_factory = mtcnn_model.onet_cnn4
    elif(config.mtcnn_model == "onet_cnn5"):
        net_factory = mtcnn_model.onet_cnn5
    elif(config.mtcnn_model == "onet_cnn6"):
        net_factory = mtcnn_model.onet_cnn6
        
    train(net_factory, prefix, end_epoch, display=display, base_lr=lr)

if __name__ == '__main__':
    path = "/home/wassimea/Desktop/wassimea/work/train_models"#sys.argv[1]
    sys.path.append(path)
    #os.chdir(path)
    print("PWD", os.getcwd)
    #config = __import__(sys.argv[2])
    config = __import__("MTCNN_config")
    x = singleton.configuration(config.config)
    config = config.config
    config = singleton.configuration._instance.config
    model_name = 'MTCNN'
    model_path = config.model_out + str(config.input_channels) + '_' + config.mtcnn_model + config.bbox_loss + '/'
    prefix = model_path
    end_epoch = config.end_epoch
    display = 10
    lr = 0.001
    train_ONet(prefix, end_epoch, display, lr)
