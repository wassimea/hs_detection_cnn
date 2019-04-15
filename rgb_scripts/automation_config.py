import sys
import os
from easydict import EasyDict as edict

config = edict()

#mod

config.mod = True
#generate_channels configs

config.gt_json = "/Data2TB/sample/annotations.json"
config.prop_json = "/Data2TB/correctly_registered/S12/train/output.json"
config.pos_out_json = "/Data2TB/chl_data/mod/train/train_pos.json"
config.pos_rgb_in = "/Data2TB/correctly_registered/S12/train/depth/"
config.pos_rgb_out = "/Data2TB/chl_data/mod/train/pos_png/"



#get_negatives configs

config.neg_rgb_in = "/Data2TB/correctly_registered/S12/train/negatives/depth/"
config.neg_rgb_out = "/Data2TB/chl_data/mod/train/neg_png/"

#get_negatives_from_pos_images config


#generate_json_negatives config

config.neg_json_out = "/Data2TB/chl_data/mod/train/train_neg.json"

#neg_to_val config

config.val_reference_folder = "/Data2TB/chl_data/rgb/val/"
config.val_source_folder = "/Data2TB/chl_data/mod/train/"
config.val_destination_folder = "/Data2TB/chl_data/mod/val/"

#remove outliers config

config.train_neg_reference_folder = "/Data2TB/chl_data/rgb/train/neg_png/"