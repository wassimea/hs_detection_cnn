#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.train_gpu = "0"
config.eval_gpu = "0"

config.BATCH_SIZE = 200
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [500,1000,1200]

config.mtcnn_model = "mobilenet"
config.model_out = "/home/wassimea/Desktop/wassimea/work/train_models/mn/"

#train configs

config.num = 44703
config.end_epoch = 2000
config.train_pos_record = "/Data2TB/chl_data/mod/train/records/pos.record"
config.train_neg_record = "/Data2TB/chl_data/mod/train/records/neg.record"
config.val_pos_record = "/Data2TB/chl_data/mod/val/records/pos.record"
config.val_neg_record = "/Data2TB/chl_data/mod/val/records/neg.record"
config.pos_radio = 2.0/6
config.neg_radio = 4.0/6

config.input_channels = 3

config.radio_cls_loss = 1.0
config.radio_bbox_loss = 1.0
config.radio_landmark_loss = 1.0

config.bbox_loss = "mse"
config.landmark_loss = "mse"

config.image_size = 48


#eval configs
config.rgb_folder = "/Data2TB/correctly_registered/S12/test/depth/"
config.display_folder = "/Data2TB/correctly_registered/S12/test/color/"
config.jsonprop = "/Data2TB/correctly_registered/S12/test/output.json"
config.jsongt = "/Data2TB/sample/annotations.json"
config.results_out = "/home/wassimea/Desktop/wassimea/work/detection/results/"
config.from_index = 600
config.to_index = 1500
config.step_index = 10
#ONLY 4D
config.d_folder = "/Data2TB/correctly_registered/S12/test/depth_normalized/"

